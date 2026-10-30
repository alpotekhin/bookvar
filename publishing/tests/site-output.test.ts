import { access, readFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import YAML from 'yaml';
import { findBrokenBuiltLinks } from '../adapter/check-built-links.js';
import { loadManifest } from '../adapter/manifest.js';
import { readingPaths } from '../adapter/reading-paths.js';

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const distDir = join(rootDir, 'site', 'dist');
const pagefindDir = join(distDir, 'pagefind');

interface PagefindResult {
  data(): Promise<{ meta: { title: string }; url: string }>;
}

interface PagefindApi {
  destroy(): Promise<void>;
  options(options: Record<string, unknown>): Promise<void>;
  search(term: string): Promise<{ results: PagefindResult[] }>;
}

let pagefind: PagefindApi;
let originalFetch: typeof globalThis.fetch;

function expectCanonicalEditionNotice(html: string) {
  expect(html.match(/aria-label="English edition status"/g)).toHaveLength(1);
  expect(html).toContain('Canonical-language chapter.');
  expect(html).toContain('Read the canonical chapter');
  expect(html).not.toContain('Partial English edition.');
  expect(html).not.toContain('This content is not available in your language yet.');
}

function expectCourseOriginalAlias(html: string, route: string) {
  expect(html).toContain(`0;url=/${route}/`);
  expect(html).toContain(`href="/${route}/"`);
  expect(html).toContain('location.search + location.hash');
  expect(html).toContain('noindex');
}

beforeAll(async () => {
  originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const url = typeof input === 'string' || input instanceof URL ? String(input) : input.url;
    if (!url.startsWith('file:')) return originalFetch(input, init);
    const contents = await readFile(fileURLToPath(url));
    return new Response(contents as unknown as BodyInit, {
      status: 200,
      headers: { 'content-type': url.endsWith('.pagefind') ? 'application/wasm' : 'application/octet-stream' }
    });
  };

  const basePath = pathToFileURL(`${pagefindDir}/`).href;
  pagefind = await import(`${basePath}pagefind.js`) as PagefindApi;
  await pagefind.options({ basePath, language: 'ru' });
});

afterAll(async () => {
  await pagefind?.destroy();
  globalThis.fetch = originalFetch;
});

describe('static handbook output', () => {
  it('publishes the licensed replacements and all configuration data in both interfaces', async () => {
    const registry=YAML.parse(await readFile(join(rootDir,'05 Источники/asset-registry.yml'),'utf8'));
    const retiredAssets=registry.assets.filter((row: { publication_status?: string }) => row.publication_status === 'retired');
    expect(retiredAssets).toHaveLength(3);
    for(const prefix of ['', 'en/']){
      const gpt=await readFile(join(distDir,prefix+'textbook/models/gpt-3/index.html'),'utf8');
      const table=[...gpt.matchAll(/<table\b[\s\S]*?<\/table>/g)].map(x=>x[0]).find(x=>x.includes('GPT-3 Small'));
      expect(table).toBeDefined();
      expect(table!.match(/<th(?:\s|>)/g)).toHaveLength(8);
      expect(table!.match(/<td(?:\s|>)/g)).toHaveLength(64);
      for(const label of ['GPT-3 Small','GPT-3 Medium','GPT-3 Large','GPT-3 XL',
        'GPT-3 2.7B','GPT-3 6.7B','GPT-3 13B','GPT-3 175B']) expect(table).toContain(label);
      expect(gpt).toContain('gpt3-demonstrations-frame-171.png');
      expect(gpt).toContain('https://jalammar.github.io/images/gpt3/09-gpt3-generating-react-code-example.gif');
      expect(gpt).not.toContain('katex-error');
      const grpo=await readFile(join(distDir,prefix+'textbook/post-training/grpo-deepseek-r1/index.html'),'utf8');
      expect(grpo).toContain('ppo-framework.png');
      expect(grpo).toContain('group-advantages.png');
      expect(grpo).toContain('/licenses/rlhf-book-MIT.txt');
      if(prefix) expectCanonicalEditionNotice(grpo);
      for(const retired of retiredAssets){
        const name=retired.asset.split('/').at(-1)!;
        expect(gpt).not.toContain(name);
        expect(grpo).not.toContain(name);
      }
    }
    for(const retired of retiredAssets)
      await expect(access(join(distDir,'assets',retired.asset))).rejects.toThrow();
  });

  it('keeps Plan-Seq-Learn attribution optional while retaining the source explanation and figures', async () => {
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + 'textbook/agents/memory-planning-orchestration/index.html'), 'utf8');
      const disclosure = html.match(/(<details[^>]*>)\s*<summary>Source text and attribution<\/summary>([\s\S]*?)<\/details>/);
      expect(disclosure).not.toBeNull();
      expect(disclosure![1].includes('open')).toBe(false);
      expect(disclosure![2].includes('Ниже сохранены')).toBe(true);
      expect(disclosure![2].includes('<blockquote>')).toBe(false);
      const start = html.indexOf('id="plan-seq-learn"');
      const end = html.indexOf('id="всегда-ли-дополнительный-планировщик-полезен"', start);
      const section = html.slice(start, end);
      for (const figure of ['method.png', 'nut-assembly.png', 'sequencing.png'])
        expect(section.includes('/planseqlearn-2026-10-26/' + figure)).toBe(true);
      for (const text of ['Planning Module: Zero-Shot High-level Planning',
        'Sequencing Module: Vision-based Plan Tracking', 'Learning Module: Efficiently Learning Local Control',
        'Stage Termination Details', 'There are several limitations of PSL'])
        expect(section.includes(text)).toBe(true);
      expect(html.includes('katex-error')).toBe(false);
      if (prefix) expectCanonicalEditionNotice(html);
    }
  });
  it('renders the three modern-route corrections in both interfaces without dropping examples', async () => {
    for (const prefix of ['', 'en/']) {
      const ffn = await readFile(join(distDir, prefix + 'textbook/modern-llm/dense-ffn/index.html'), 'utf8');
      const long = await readFile(join(distDir, prefix + 'textbook/modern-llm/long-context/index.html'), 'utf8');
      const distributed = await readFile(join(distDir, prefix + 'textbook/pretraining/distributed-systems/index.html'), 'utf8');
      expect(ffn.includes('Краткие формулы собраны отдельно')).toBe(false);
      expect(long.includes('Все четыре блока NumPy')).toBe(false);
      expect(long).toContain('Примеры NumPy в главе можно');
      const prose = distributed.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ');
      expect(prose.includes('Предыдущая глава о масштабировании')).toBe(false);
      expect(distributed).toContain('Prerequisites');
      for (const route of ['modern-llm/transformer-shapes-resources',
        'ml-systems/transformer-moe-arithmetic', 'pretraining/scaling-laws'])
        expect(distributed).toContain('href="/' + prefix + 'textbook/' + route + '/"');
      for (const [html, count] of [[ffn, 3], [long, 5], [distributed, 3]] as const) {
        expect(html.match(/<pre data-language=/g)).toHaveLength(count);
        expect(html.includes('katex-error')).toBe(false);
        if (prefix) expectCanonicalEditionNotice(html);
      }
    }
  });
  it('keeps canonical-language textbook navigation in EN while labeling the body honestly', async () => {
    const source = await readFile(join(rootDir,
      'site/src/content/docs/generated/en/textbook/evaluation/models-contamination.md'), 'utf8');
    const data = YAML.parse(source.match(/^---\n([\s\S]*?)\n---/)![1]);
    expect(data.translationStatus).toBe('canonical');
    expect(data.canonicalHref).toBe('/textbook/evaluation/models-contamination/');
    expect(data.translationReviewed).toBeUndefined();
    const html = await readFile(join(distDir, 'en/textbook/evaluation/models-contamination/index.html'), 'utf8');
    expectCanonicalEditionNotice(html);
    for (const route of ['inference/benchmarking-slo', 'agents/evaluation',
      'pretraining/dedup-pii-corpus-quality', 'inference/decoding']) {
      expect(html).toContain('href="/en/textbook/' + route + '/"');
      expect(html).not.toContain('href="/textbook/' + route + '/"');
    }
    expect(html).toContain('href="/textbook/evaluation/models-contamination/"');
    expect(html).toContain('Две модели правильно ответили');
    expect(html).toContain('bert-score-recall.png');
  });
  it('shows one edition notice per textbook page, or none for current/native English content', async () => {
    const manifest = loadManifest(join(rootDir, 'publishing/navigation.yml'));
    const textbook = manifest.sections.find(section => section.id === 'textbook')!;
    for (const page of textbook.pages) {
      const file = page.route.replace(/\/index$/, '') + '/index.html';
      const raw = await readFile(join(rootDir, page.source), 'utf8');
      const canonical = YAML.parse(raw.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '') ?? {};
      const root = await readFile(join(distDir, file), 'utf8');
      const english = await readFile(join(distDir, 'en', file), 'utf8');
      expect(root, page.route).not.toContain('aria-label="English edition status"');
      expect(english, page.route).not.toContain('This content is not available in your language yet.');
      const rootChapterLinks = [...english.matchAll(
        /<a\b[^>]*\bhref="(\/textbook\/[^\"]*)"[^>]*>([\s\S]*?)<\/a>/g)]
        .filter(match => match[2].replace(/<[^>]*>/g, '').trim() !== 'Read the canonical chapter')
        .map(match => ({ href: match[1], text: match[2].replace(/<[^>]*>/g, '').trim() }));
      expect(rootChapterLinks, page.route + ': unexpected locale switch').toEqual([]);
      if (!page.sourceEn && canonical.source_language !== 'en') {
        expectCanonicalEditionNotice(english);
        const media = (text: string) => [...text.matchAll(
          /<(?:img|video|source)\b[^>]*\bsrc="([^"]+)"/g)].map(match => match[1]);
        const programs = (text: string) => [...text.matchAll(/<pre\b[^>]*>([\s\S]*?)<\/pre>/g)]
          .map(match => match[1].replace(/<[^>]*>/g, ''));
        expect(media(english), page.route + ': original figures/media').toEqual(media(root));
        expect(programs(english), page.route + ': published code').toEqual(programs(root));
        expect(english.match(/<math\b[\s\S]*?<\/math>/g) ?? [], page.route + ': formulas')
          .toEqual(root.match(/<math\b[\s\S]*?<\/math>/g) ?? []);
      } else if (!page.sourceEn) {
        expect(english, page.route).not.toContain('aria-label="English edition status"');
      } else {
        const text = await readFile(join(rootDir, page.sourceEn), 'utf8');
        const translation = YAML.parse(text.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '') ?? {};
        const current = translation.translation_status === 'current'
          && translation.canonical_revision === createHash('sha256').update(raw).digest('hex');
        expect(english.match(/aria-label="English edition status"/g) ?? [], page.route)
          .toHaveLength(current ? 0 : 1);
        if (!current) expect(english, page.route).toContain('Partial English edition.');
      }
    }
    const other = manifest.sections.find(section => section.id === 'models')!.pages
      .find(page => !page.sourceEn)!;
    const outside = await readFile(join(distDir, 'en', other.route.replace(/\/index$/, ''), 'index.html'), 'utf8');
    expect(outside).toContain('This content is not available in your language yet.');
  });
  it('renders the complete nanoGPT worked reference in both existing practice routes', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'practice/20-sobrat-yazykovuyu-model-s-nulya/index.html'), 'utf8');
      const text = html.replace(/<[^>]+>/g, '');
      expect(html).toContain('id="nanogpt-worked-reference"');
      expect(html).toContain('nanogpt-2026-10-30/loss-original.png');
      expect(html).toContain('nanogpt-3adf61e1-MIT.txt');
      for (const fragment of ['96 training targets', '110,392-byte', '48 official tests',
        'Checkpoint contents and update boundaries', 'Original nanoGPT excerpt',
        'optimizer.zero_grad(set_to_none=True)', 'dtype']) {
        expect(text).toContain(fragment);
      }
      expect(html).not.toContain('katex-error');
    }
  });
  it('does not leak temporary code-protection markers into any published page', async () => {
    const files = (await readdir(distDir, { recursive: true })).filter(file => file.endsWith('index.html'));
    const leaked: string[] = [];
    for (const file of files) {
      if (/BOOKVARPROTECTEDCODE\d+TOKEN/.test(await readFile(join(distDir, file), 'utf8'))) leaked.push(file);
    }
    expect(leaked).toEqual([]);
  });
  it('keeps all eight complete-Transformer code blocks in both locales', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/transformer/full-transformer/index.html'), 'utf8');
      expect(html.match(/<pre\b/g)).toHaveLength(8);
      expect(html).toContain('id="transformer-dropout"');
      expect(html.replace(/<[^>]+>/g,'')).toContain('A surviving weight can exceed one');
      for (const name of ['TinyTranslation', 'TinyCausalLM', 'class EncoderBlock', 'class DecoderBlock'])
        expect(html.replace(/<[^>]+>/g, '')).toContain(name);
    }
  });
  it('renders complete BLEU, chrF and original BERTScore reasoning in both locale routes', async () => {
    for (const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale,'textbook/evaluation/models-contamination/index.html'),'utf8');
      for (const anchor of ['reference-based-text-evaluation','bleu-reference-overlap',
        'chrf-character-overlap','bertscore-token-alignment'])
        expect(html).toContain('id="'+anchor+'"');
      for (const value of ['bleu_from_tokens','Corpus BLEU','weighted_recall',
        'bert-score-recall.png','bertscore-19e7f551-MIT.txt','53.7285'])
        expect(html).toContain(value);
      expect(html.match(/<pre\b/g)).toHaveLength(13);
      expect(html).not.toContain('katex-error');
      expect(html).not.toContain('<p>![');
      if(locale)expectCanonicalEditionNotice(html);
    }
  });
  it('renders Coconut inside TTC, including the native diagram, complete example and mixed-language fallback', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/reasoning/test-time-compute/index.html'), 'utf8');
      expect(html).toContain('id="coconut-continuous-thought"');
      expect(html).toContain('coconut-2026-10-29/coconut.png');
      expect(html).toContain('coconut-27273cb8-MIT.txt');
      expect(html).toContain('coconut_linear_step');
      expect(html).toContain('coconut_stage');
      expect(html.match(/<pre\b/g)).toHaveLength(7);
      for (const anchor of ['tree-of-thoughts', 'tot-creative-writing', 'tot-crossword-dfs'])
        expect(html).toContain('id="' + anchor + '"');
      for (const file of ['tot-creative-search.png', 'tot-creative-results.png', 'tot-crossword-dfs.png'])
        expect(html).toContain('tot-reconciliation-2026-10-30/' + file);
      expect(html).not.toContain('katex-error');
      if (locale) expectCanonicalEditionNotice(html);
    }
  });
  it('serves all declared source-native English pages without pretending they need translation', async () => {
    const manifest = loadManifest(join(rootDir, 'publishing/navigation.yml'));
    const native = [];
    for (const page of manifest.sections.flatMap(section => section.pages)) {
      const raw = await readFile(join(rootDir, page.source), 'utf8');
      const metadata = YAML.parse(raw.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '') ?? {};
      if (!page.sourceEn && metadata.source_language === 'en') native.push(page);
    }
    expect(native).toHaveLength(16);
    for (const page of native) {
      const file = page.route.replace(/\/index$/, '') + '/index.html';
      const canonical = await readFile(join(distDir, file), 'utf8');
      const english = await readFile(join(distDir, 'en', file), 'utf8');
      expect(english, page.route).not.toContain('This content is not available in your language yet.');
      expect(english, page.route).not.toContain('Canonical-language chapter.');
      expect(english, page.route).not.toContain('Partial English edition.');
      expect(english.match(/<img\b[^>]*\bsrc=/g) ?? [], page.route)
        .toHaveLength((canonical.match(/<img\b[^>]*\bsrc=/g) ?? []).length);
      expect(english.match(/<pre\b/g) ?? [], page.route)
        .toHaveLength((canonical.match(/<pre\b/g) ?? []).length);
      expect(english.match(/class="katex-display"/g) ?? [], page.route)
        .toHaveLength((canonical.match(/class="katex-display"/g) ?? []).length);
      expect(english, page.route).not.toContain('katex-error');
    }
  });
  it('publishes the same complete PEFT chapter, ten original figures and eighteen examples in both locale routes', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/post-training/fine-tuning-peft/index.html'), 'utf8');
      expect(html.match(/<h1(?:\s|>)/g)).toHaveLength(1);
      expect(html.match(/<img\b[^>]*\bsrc=/g)).toHaveLength(10);
      expect(html.match(/<pre\b/g)).toHaveLength(18);
      for (const name of ['houlsby-adapter.png', 'lora-animated.gif', 'lora-merge.png',
        'qlora-training-original.png', 'prompt-tuning.png', 'prefix-tuning.png',
        'asapp-overview.png', 'mosbach-stability.png',
        'l2sp-target-comparison.png', 'l2sp-retention.png'])
        expect(html).toContain(name);
      for (const id of ['choosing-trainable-parameters', 'bottleneck-adapters', 'lora-update',
        'merging-and-serving', 'qlora-training', 'soft-prompts', 'prefix-keys-values',
        'small-data-fine-tuning', 'fine-tuning-bias-correction', 'fine-tuning-reinitialization',
        'fine-tuning-update-budget', 'intermediate-task-transfer',
        'retaining-capabilities', 'fine-tuning-replay', 'l2-sp',
        'elastic-weight-consolidation', 'fisher-for-ewc'])
        expect(html).toContain('id="' + id + '"');
      expect(html).toContain('href="/licenses/peft-fee923eb-Apache-2.0.txt"');
      expect(html).toContain('href="/licenses/asapp-finetuning-0aa4f4e-LICENSE.txt"');
      expect(html).toContain('href="/licenses/mosbach-finetuning-3cf27e8-Apache-2.0.txt"');
      expect(html).toContain('CC BY-NC-SA 4.0');
      expect(html).not.toContain('katex-error');
      expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders masked diffusion with four source figures, executable programs and module navigation', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/pretraining/masked-diffusion-language-models/index.html'), 'utf8');
      for (const anchor of ['masking-process', 'weighted-mask-loss', 'reverse-transition', 'complete-toy-sampler',
        'llada-training-and-selection', 'block-diffusion-and-kv', 'remasking-and-refinement', 'likelihood-and-serving'])
        expect(html).toContain(`id="${anchor}"`);
      for (const name of ['mdlm-overview.png', 'block-diffusion-overview.svg', 'remasking-overview.svg', 'remasking-loop.svg'])
        expect(html).toContain('masked-diffusion-2026-10-29/' + name);
      expect(html.match(/<pre\b/g)).toHaveLength(5);
      expect(html.match(/class="katex-display"/g)).toHaveLength(8);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('CC BY 4.0');
      expect(html).toContain('textbook/pretraining/mae-masked-autoencoders/');
      if (locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders MAE source figures, complete programs and source-native English rendering',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/mae-masked-autoencoders/index.html'),'utf8');
      for(const anchor of ['asymmetric-mae','patch-layout','mask-and-restore','masked-pixel-loss',
        'normalized-targets','mae-training-step','reading-reconstructions','mae-cost-and-evaluation'])
        expect(html).toContain(`id="${anchor}"`);
      for(const name of ['architecture.png','pixel-reconstruction.png','gan-comparison.png'])
        expect(html).toContain('mae-2026-10-28/'+name);
      expect(html.match(/<pre\b/g)).toHaveLength(6);
      expect(html.match(/class="katex-display"/g)).toHaveLength(2);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('/licenses/mae-efb2a80-CC-BY-NC-4.0.txt');
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders MoCo queue, EMA, source diagrams and source-native English rendering',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/moco-queue-momentum/index.html'),'utf8');
      for(const anchor of ['dictionary','momentum-encoder','fifo-queue','moco-loss',
        'queue-snapshot','complete-step','distributed-shuffle','cost-and-versions'])
        expect(html).toContain(`id="${anchor}"`);
      expect(html).toContain('moco-2026-10-28/moco-dictionary.png');
      expect(html).toContain('simclr-2026-10-28/framework.png');
      expect(html.match(/<pre\b/g)).toHaveLength(6);
      expect(html.match(/class="katex-display"/g)).toHaveLength(3);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('/licenses/moco-132aea9-MIT.txt');
      expect(html).toContain('/licenses/moco-aea97d4-CC-BY-NC-4.0.txt');
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders SimCLR source diagrams, programs, math and source-native English rendering',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/simclr-representations/index.html'),'utf8');
      for(const anchor of ['two-view-encoder','augmentation-task','positive-index','nt-xent',
        'shared-training-step','collapse-and-negatives','projection-versus-representation','linear-evaluation'])
        expect(html).toContain(`id="${anchor}"`);
      for(const name of ['framework.png','crop-views.png','augmentations.png','projection-head.png'])
        expect(html).toContain('simclr-2026-10-28/'+name);
      expect(html.match(/<pre\b/g)).toHaveLength(5);
      expect(html.match(/class="katex-display"/g)).toHaveLength(2);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('/licenses/uva-c7d9e18-MIT.txt');
      expect(html).toContain('/licenses/simclr-PMLR-CC-BY-4.0.txt');
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders semi-supervised mechanisms, unchanged original figures and source-native English rendering',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/semi-supervised-fixmatch/index.html'),'utf8');
      for(const anchor of ['pseudo-labels','consistency','fixmatch','training-step','confirmation-bias',
        'label-propagation','published-results','cost-accounting','self-supervised-boundary'])
        expect(html).toContain(`id="${anchor}"`);
      for(const name of ['fixmatch-diagram.png','circles-raw.png','circles-spread.png'])
        expect(html).toContain('semi-supervised-2026-10-28/'+name);
      expect(html.match(/<pre\b/g)).toHaveLength(14);
      expect(html.match(/class="katex-display"/g)).toHaveLength(22);
      for(const anchor of ['noisy-student','mixtext','dai-le-sequence-pretraining','ssl-structural-assumptions'])
        expect(html).toContain(`id="${anchor}"`);
      for(const name of ['tmix-original.png','mixtext-original.png'])
        expect(html).toContain('semi-supervised-extensions-2026-10-29/'+name);
      for(const anchor of ['virtual-adversarial-training','uda','training-signal-annealing'])
        expect(html).toContain(`id="${anchor}"`);
      for(const name of ['vat.gif','uda-fixmatch.png'])
        expect(html).toContain('consistency-training-2026-10-29/'+name);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('/licenses/fixmatch-d4985a1-Apache-2.0.txt');
      expect(html).toContain('/licenses/semi-supervised-sklearn-BSD-3-Clause.txt');
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders embedding adversarial training, FreeLB and SMART in both locales',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/deployment/robustness/index.html'),'utf8');
      for(const anchor of ['embedding-adversarial-training','freelb','smart'])
        expect(html).toContain(`id="${anchor}"`);
      expect(html).toContain('consistency-training-2026-10-29/smart-boundaries.png');
      expect(html).toContain('0.8036');
      expect(html).toContain('0.7892');
      expect(html.match(/<pre\b/g)).toHaveLength(3);
      expect(html.match(/class="katex-display"/g)).toHaveLength(12);
      expect(html).not.toContain('katex-error');
    }
  });
  it('renders geometric selection with original figures, math, code and source-native English rendering',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/geometric-coresets/index.html'),'utf8');
      for(const anchor of ['coverage-objective','farthest-first','approximation-guarantee',
        'implementation','representation-and-metric','outliers-and-labels',
        'coreset-versus-centres','herding']) expect(html).toContain(`id="${anchor}"`);
      expect(html).toContain('geometric-coresets-2026-10-28/erickson-farthest-first.png');
      expect(html).toContain('clustering_quiz_kmeans_not_scaled.svg');
      expect(html).toContain('clustering_quiz_kmeans_scaled.svg');
      expect(html.match(/<pre\b/g)).toHaveLength(6);
      expect(html.match(/class="katex-display"/g)).toHaveLength(10);
      expect(html).not.toContain('katex-error');
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders active-learning mechanisms, original images, complete programs and source-native English rendering',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/active-learning/index.html'),'utf8');
      for(const anchor of ['annotation-loop','uncertainty','committee','expected-gradient',
        'executable-loop','batch-and-diversity','evaluation-and-cost']) expect(html).toContain(`id="${anchor}"`);
      for(const figure of ['active-learning.png','motivating-example.png'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*active-learning-2026-10-28/'+figure.replace('.','\\.')));
      expect(html.match(/<pre\b/g)).toHaveLength(5);
      expect(html).toContain('/licenses/modal-bba6f6fd-MIT.txt');
      expect(html.match(/class="katex-display"/g)).toHaveLength(8);
      expect(html).not.toContain('katex-error');
      expect(html).toContain('oracle_calls');
      expect(html).toContain('consensus entropy');
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders labelled selection and Confident Learning with four source figures and eight complete programs',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/pretraining/data-selection-scores/index.html'),'utf8');
      for(const anchor of ['selection-problem','el2n','grand','forgetting','stable-identifiers',
        'reproduction','selection-controls','confident-learning','selection-cost']) expect(html).toContain(`id="${anchor}"`);
      for(const figure of ['joost_train_input_norm_vs_grad_norm_sorted.png','figure11_repro_cifar10_hlb.svg'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*data-selection-2026-10-28/'+figure.replace('.','\\.')));
      for(const figure of ['figure1-process.png','figure2-joint-estimation.png'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*confident-learning-2026-10-29/'+figure.replace('.','\\.')));
      expect(html.match(/<pre\b/g)).toHaveLength(8);
      expect(html.match(/class="katex-display"/g)).toHaveLength(17);
      expect(html).not.toContain('katex-error');
      for(const fact of ['never_correct','score_by_id','shaded bands','Apache-2.0','CC BY-NC-SA 4.0'])
        expect(html).toContain(fact);
      for(const license of ['data-diet-aee462ea-Apache-2.0.txt','data-diet-reproduction-fda604dc-Apache-2.0.txt'])
        await access(join(distDir,'licenses',license));
      expect(html).toContain('/licenses/confident-learning-JAIR-v1.html');
      await access(join(distDir,'licenses/confident-learning-JAIR-v1.html'));
      if(locale) expect(html).not.toContain('This content is not available in your language yet.');
    }
  });
  it('renders original GaLoRE figures, the three memory examples and explicit EN fallback',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/neural-networks/optimizers-schedules/index.html'),'utf8');
      for(const anchor of ['memory-efficient-optimizers','adafactor-factor-the-second-moment-not-the-weight-matrix',
        '8-bit-optimizer-states-fewer-bytes-for-the-same-coordinates',
        'galore-run-adam-in-a-smaller-gradient-space','account-for-the-remaining-tensors'])
        expect(html).toContain(`id="${anchor}"`);
      for(const figure of ['galore_code_box.png','subspace_learning.png','memory_breakdown.svg'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*galore-2026-10-28/'+figure.replace('.','\\.')));
      expect(html).toContain('/licenses/galore-2cc66f88-Apache-2.0.txt');
      expect(html).toContain('id="galore-memory-breakdown"');
      expect(html).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(html.match(/<pre\b/g)).toHaveLength(9);
      for (const anchor of ['agc-unitwise','autoclip'])
        expect(html).toContain('id="'+anchor+'"');
      for (const name of ['nfnet-agc-ablation.png','autoclip-mi.gif'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*gradient-clipping-2026-10-29/'+name.replace('.','\\.')));
      expect(html).toContain('/licenses/autoclip-dfb622c3-MIT.txt');
      expect(html).not.toContain('katex-error');
      expect(html).not.toContain('LoRA, QLoRA и PEFT');
      if(locale) expectCanonicalEditionNotice(html);
    }
  });
  it('renders the original TensorFlow source and the actually checked SavedModel-to-Lite boundary',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/deployment/ml-workflow/index.html'),'utf8');
      for(const anchor of ['tensorflow-saved-graphs-variables-and-callable-signatures',
        'what-survives-saving-and-loading','named-endpoints-and-the-exported-state',
        'conversion-to-a-smaller-runtime-is-another-checked-boundary','keep-a-tested-migration-path'])
        expect(html).toContain(`id="${anchor}"`);
      for(const fact of ['two-layer-network.png','CustomModule','checked_scale','4.98186',
        'BUILTIN_WITHOUT_DEFAULT_DELEGATES','flattened signature callable']) expect(html).toContain(fact);
      expect(html).toContain('/licenses/tensorflow-docs-9beed83c-Apache-2.0.txt');
      expect(html).not.toContain('katex-error');
    }
    expect(await readFile(join(distDir,'licenses/tensorflow-docs-9beed83c-Apache-2.0.txt'),'utf8'))
      .toBe(await readFile(join(rootDir,'site/public/licenses/tensorflow-docs-9beed83c-Apache-2.0.txt'),'utf8'));
  });
  it('publishes the worked model-export boundary and full original provider figure in both routes',async()=>{
    for(const locale of ['', 'en/']){
      const html=await readFile(join(distDir,locale,'textbook/deployment/ml-workflow/index.html'),'utf8');
      expect(html).toContain('id="exporting-a-model-does-not-export-the-application"');
      expect(html).toContain('id="which-provider-executes-each-part"');
      expect(html).toContain('onnx-runtime-providers.png');
      for(const fact of ['MissingKernel','CPUExecutionProvider','Batch 9','1454×910']) expect(html).toContain(fact);
      expect(html).toContain('/licenses/onnxruntime-e3d000e3-MIT.txt');
      expect(html).not.toContain('katex-error');
    }
    expect(await readFile(join(distDir,'licenses/onnxruntime-e3d000e3-MIT.txt'),'utf8'))
      .toBe(await readFile(join(rootDir,'site/public/licenses/onnxruntime-e3d000e3-MIT.txt'),'utf8'));
  });
  it('renders the camera-trap memory, deadline and disjoint-state energy calculation', async () => {
    for(const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale,'practice/18a-architecture-selection/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      expect(html).toContain('id="a-camera-trap-memory-event-latency-and-battery-energy"');
      for(const value of ['65.045056','75 ms','865.425','43.27125','50.325','8.64 kWh',
        'same MAC/FLOP convention','class counts alone do not prove']) expect(text).toContain(value);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(4);
      expect(html).toContain('harvard-architecture-selection.svg');
      expect(html).toContain('harvard-architecture-budgets.svg');
      expect(html).toContain('/examples/architecture-lab-source-checks-2026-10-26.json');
      expect(html).not.toContain('katex-error');
    }
  });

  it('renders complete Winograd examples, native figure and original downloads in both routes', async () => {
    for(const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale,
        'textbook/neural-networks/cnn-lenet-resnet/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      expect(html).toMatch(/<img\b[^>]*src="[^"]*\/winograd-2026-10-27\/dolz-workflow\.png"/);
      for(const fact of ['Four products recover two outputs', 'Channels become sixteen',
        '2049', '0.5 instead of 1', 'not a measured speedup', 'torch.bmm', 'CC BY 4.0'])
        expect(text).toContain(fact);
      expect(html).toContain('href="/examples/wincnn-c2bf6305.py"');
      expect(html).toContain('href="/licenses/wincnn-c2bf6305-Apache-2.0.txt"');
      expect(html).not.toContain('katex-error');
      if(locale) expectCanonicalEditionNotice(html);
    }
    expect(await readFile(join(distDir,'examples/wincnn-c2bf6305.py')))
      .toEqual(await readFile(join(rootDir,'site/public/examples/wincnn-c2bf6305.py')));
  });

  it('renders the original Harvard window diagram with the corrected orientation in both routes', async () => {
    for(const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale,
        'textbook/neural-networks/cnn-lenet-resnet/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      expect(html).toContain('id="two-channels-window-rows-and-filter-vector"');
      expect(html).toMatch(/<img\b[^>]*src="[^"]*\/nn-native-2026-10-27\/harvard-window-rows\.svg"/);
      for(const fact of ['356', '392', '464', '500', 'each channel',
        'This explanatory caption corrects', 'No timing measurement', 'CC BY-NC-SA 4.0'])
        expect(text.toLowerCase()).toContain(fact.toLowerCase());
      expect(html).not.toContain('katex-error');
    }
  });

  it('renders rooted/all-rank destinations and complete variable-split routing', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale,
        'textbook/pretraining/processes-collectives-ddp/index.html'), 'utf8');
      const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
      expect(html).toContain('id="collective-destinations-gather-scatter-all-to-all"');
      for (const name of ['scatter.png', 'gather.png', 'all_gather.png'])
        expect(html).toMatch(new RegExp('<img\\b[^>]*src="[^"]*/collective-destinations-2026-10-27/'
          + name.replaceAll('.', '\\.') + '"'));
      for (const fact of ['input_split_sizes', 'output_split_sizes', 'sends five',
        'receives six', 'CPU/Gloo', 'experimental']) expect(text).toContain(fact);
      expect(html).toContain('/licenses/pytorch-tutorials-c8ccbbf-BSD-3-Clause.txt');
      expect(html).not.toContain('katex-error');
    }
  });

  it('renders indexed gather/scatter, original SVG, gradients and both API contracts', async () => {
    const html = await readFile(join(distDir,
      'textbook/ml-systems/gpu-kernels-triton/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="indexed-gather-scatter-and-gradients"');
    expect(html).toMatch(/<img\b[^>]*\bsrc="[^"]*\/scatter-add\.svg"/);
    for (const phrase of ['GatherElements-13', 'Scatter-add combines values',
      'initial.scatter_add', 'fp32_access_units', 'full one-to-one permutation',
      'not measured HBM traffic']) expect(text).toContain(phrase);
    expect(html).toContain('/licenses/pytorch-scatter-f514c10-MIT.txt');
    expect(html).not.toContain('katex-error');
  });

  it('renders the original reduction animation, worked tree and copied CUDA helper', async () => {
    const html = await readFile(join(distDir,
      'textbook/ml-systems/gpu-kernels-triton/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="local-reduction-tree-work-depth-synchronization"');
    expect(html).toMatch(/<img\b[^>]*\bsrc="[^"]*\/binomial-tree\.gif"/);
    for (const phrase of ['reduction_trace', 'Independent additions', 'cg::sync(cta)',
      'warpReduceSum', '__shfl_down_sync', 'all 32 lanes to participate',
      'not a complete kernel or an executed CUDA test', 'Flogr']) expect(text).toContain(phrase);
    expect(html).toContain('/licenses/nvidia-cuda-samples-5443602-BSD-3-Clause.txt');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(45);
    expect(html).not.toContain('katex-error');
  });

  it('renders the Harvard Transformer figure joined to a bare DOI without changing its source', async () => {
    const source=await readFile(join(rootDir,
      '05 Источники/Courses/Harvard ML Systems/vol1/nn_architectures.md'));
    expect(createHash('sha256').update(source).digest('hex'))
      .toBe('52c77d7f6d386e97af65decf4f55523be785930571759fe5a906069ad13fcfac');
    const route = 'sources/courses/harvard-ml-systems/vol1/network-architectures';
    const html=await readFile(join(distDir,route,'index.html'),'utf8');
    const main=html.slice(html.indexOf('<main'),html.indexOf('</main>'));
    expect(main.match(/<img\b[^>]*\bsrc="/g)).toHaveLength(24);
    expect(main).toMatch(/<img\b[^>]*\bsrc="[^"]*8f27999af14450c9623159ca4a3f23e5463a4c45\.svg"/);
    expect(main).toContain('href="https://doi.org/10.18653/v1/w19-4828"');
    expect(main).not.toContain('w19-4828.%5B!%5B');
    expectCourseOriginalAlias(await readFile(join(distDir,'en',route,'index.html'),'utf8'), route);
  });

  it('renders recurrent input projections, layer dependencies and the complete original D2L figure', async () => {
    const html=await readFile(join(distDir,'textbook/recurrent-networks/rnn-bptt/index.html'),'utf8');
    for(const id of ['parallel-work-inside-a-recurrent-network','full-sequence-chunks-and-a-live-stream',
      'several-layers-form-a-dependency-grid','weight-gradients-can-be-reduced-across-time'])
      expect(html).toContain('id="'+id+'"');
    expect(html).toMatch(/<img[^>]+src="[^"]*\/d2l-deep-rnn\.svg"/);
    expect(html.match(/<img\b[^>]+src="\/assets\//g)).toHaveLength(6);
    expect(html.match(/<pre\b/g)).toHaveLength(7);
    for(const fact of ['62.5 MiB','1.25 MiB','Original English excerpt','projection chunk:',
      'states and all gradients match','not a measured twofold GPU speedup','CC BY-SA 4.0'])
      expect(html).toContain(fact);
    expect(html).not.toContain('katex-error');
  });
  it('renders the DDP overlap explanation, original bucketing figure and stream examples', async () => {
    const html = await readFile(join(distDir,
      'textbook/pretraining/processes-collectives-ddp/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="ddp-overlap-readiness-streams-compilation"');
    expect(html).toContain('gradient-buckets.svg');
    for (const phrase of ['ordered_bucket_schedule', 'DDPOptimizer', 'record_stream',
      'whole_backward', 'split_backward', 'collective order',
      'the exposed tail is 5 ms', 'not an NCCL simulator', 'not an executed test'])
      expect(text).toContain(phrase);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(15);
    expect(html).toContain('/textbook/ml-systems/gpu-kernels-triton/#torch-compile-graphs-guards-recompilation');
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('<p>![[');
  });
  it('renders the Dynamo source explanation, original diagram and three frontend experiments', async () => {
    const html = await readFile(join(distDir,
      'textbook/ml-systems/gpu-kernels-triton/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="torch-compile-graphs-guards-recompilation"');
    expect(html).toContain('torch-dynamo-pipeline.png');
    for (const value of ['compile_records', 'piece_records', 'dynamic_records',
      'strict_boundary', 'fullgraph=True', 'AOTDispatcher',
      'not kernel launches', 'not AOTAutograd or Inductor code generation']) expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(45);
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('dynamo-summary.png');
  });
  it('renders source tracing failures and explicit export control flow', async () => {
    const html = await readFile(join(distDir,
      'textbook/ml-systems/gpu-kernels-triton/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="tracing-a-data-dependent-branch"');
    for (const value of ['Tracing Silent Failure',
      'branch_cases', 'trace_nodes', 'check_inputs', 'compiled_branch',
      'fresh compiler cache', 'torch.compiler.reset', 'Exporting an explicit conditional',
      'M_old', 'M_new', 'exported_branch', 'higher_order.cond', '0.841471',
      '0.540302', 'prototype features', 'not deployment'])
      expect(text).toContain(value);
    expect(html).toContain('/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(45);
    expect(html).not.toContain('katex-error');
  });
  it('renders the integrated CUDA Graphs explanation, original figures and source code', async () => {
    const html = await readFile(join(distDir,
      'textbook/ml-systems/gpu-kernels-triton/index.html'), 'utf8');
    for (const name of ['launch-vs-replay.png', 'mask-rcnn-before.png', 'mask-rcnn-after.png'])
      expect(html).toContain(name);
    for (const phrase of ['cuda-graphs-capture-replay', 'Why CUDA Graphs?',
      'replay_fixed_buffer', 'launch_bound_finish', 'static_input.copy_',
      'Memory pools are part of the cost', '31.662 ms', '6.078 ms',
      'not yet supported by the Inductor backend', 'not measurements of CUDA'])
      expect(html).toContain(phrase);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(45);
    expect(html).toContain('/licenses/pytorch-v2.14.1-LICENSE.txt');
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('<p>![[');
  });
  it('renders Harvard slide math, native SVGs and separate original speaker notes', async () => {
    const records = JSON.parse(await readFile(join(rootDir,
      '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json'), 'utf8'));
    const slides = records.filter((row: {kind: string}) => row.kind === 'slides');
    expect(slides).toHaveLength(35);
    for (const row of slides) {
      const html = await readFile(join(distDir, row.route, 'index.html'), 'utf8');
      expect(html, row.source).not.toContain('katex-error');
      expect(html, row.source).not.toContain('<p>![[');
      expect(html, row.source).toContain('Original speaker notes (TeX)');
      expect(html, row.source).toContain('Complete original Beamer source');
      expect(html, row.source).toContain('Original slide 1');
    }
    const edge = await readFile(join(distDir,
      'sources/courses/harvard-ml-systems/slides/vol2/11-edge-intelligence/index.html'), 'utf8');
    for (const figure of ['cloud-vs-edge-learning.svg', 'training-memory-amplifier.svg',
      'fedavg-protocol.svg', 'edge-device-spectrum.svg', 'adaptation-strategies.svg',
      'federated-systems-challenges.svg']) expect(edge).toContain(figure);
    expect(edge).toContain('Original SVG, copied without redrawing or cropping.');
    expect(edge).toContain('Source figure limitation');
    expect(edge).toContain('cannot be read as additive memory amounts');
    expect(edge).toContain('θ');
    expect(edge).toContain('<table>');
    // Lecture instructions belong to collapsed code, not narrative paragraphs.
    expect(edge).not.toMatch(/<p>[^<]*%[^<]*NARRATE:/);
    const architectures = await readFile(join(distDir,
      'sources/courses/harvard-ml-systems/slides/vol1/06-nn-architectures/index.html'), 'utf8');
    expect(architectures).toContain('Source arithmetic mismatch');
    expect(architectures).toContain('86,973,087,744');
    expect(architectures).toContain('356,241,767,399,424');
  });
  it('keeps the reading sequence separate from the common course library', async () => {
    const html=await readFile(join(distDir,'textbook/index.html'),'utf8');
    expect(html.match(/<h3\b/g)).toHaveLength(22);
    for(const part of ['Основной курс','Углублённые модули','III. Нейронные сети','V. Transformer, BERT и GPT','Рекомендательные системы'])
      expect(html).toContain(part);
    for(const alias of ['iv-нейронные-сети','iii-рекомендательные-системы','другие-режимы'])
      expect(html).toContain(`id="${alias}"`);
    expect(html).toContain('/sources/kursy/');
    expect(html).not.toContain('/sources/courses/stanford-cs336-spring-2026/');
    expect(html).not.toContain('/sources/courses/berkeley-advanced-llm-agents-spring-2025/');
    const catalog=await readFile(join(distDir,'sources/kursy/index.html'),'utf8');
    expect(catalog).toContain('/sources/courses/stanford-cs336-spring-2026/');
    expect(catalog).toContain('/sources/courses/berkeley-advanced-llm-agents-spring-2025/');
    const svm=await readFile(join(distDir,'textbook/classical-ml/kernels-svm-rff/index.html'),'utf8');
    expect(svm).not.toContain('>08 Gaussian mixture и EM</a>');
    expect(svm).toMatch(/<a\b(?=[^>]*\brel="next")(?=[^>]*\bhref="\/textbook\/#)[^>]*>/);
  });
  it('publishes previous and next buttons for all 152 chapters within their reading paths', async () => {
    const book = loadManifest(join(rootDir, 'publishing/navigation.yml')).sections.find(s => s.id === 'textbook')!;
    const publishedRoute = (route: string) => route.replace(/\/index$/, '');
    let checked = 0;
    for (const path of readingPaths(book)) for (const [i, route] of path.routes.entries()) {
      const html = await readFile(join(distDir, publishedRoute(route), 'index.html'), 'utf8');
      const anchors = html.match(/<a\b[^>]*>/g) ?? [];
      const prev = path.routes[i - 1] ? '/' + publishedRoute(path.routes[i - 1]) + '/'
        : path.id === 'main' ? '/textbook/how-to-use/' : '/textbook/#углублённые-модули';
      const next = path.routes[i + 1] ? '/' + publishedRoute(path.routes[i + 1]) + '/' : '/textbook/#углублённые-модули';
      for (const [rel, expected] of [['prev', prev], ['next', next]]) {
        const button = anchors.find(a => a.includes(`rel="${rel}"`));
        expect(button, `${route}: ${rel} button`).toBeTruthy();
        expect(decodeURI(button!.match(/href="([^"]+)"/)![1]), `${route}: ${rel}`).toBe(expected);
      }
      checked++;
    }
    expect(checked).toBe(152);
  });
  it('renders AE/VAE derivations, original course code, fourteen examples and six complete figures', async () => {
    const html = await readFile(join(distDir, 'textbook/neural-networks/autoencoder-vae/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(14);
    for (const name of ['weng-autoencoder.png', 'uva-autoencoder.svg', 'transpose-convolution.gif',
      'weng-vae-graphical-model.png', 'vae-blocks.png', 'weng-reparameterization.png']) expect(html).toContain(name);
    for (const phrase of ['class Encoder', 'class Decoder', 'class VAE', 'def loss_function',
      'posterior_gap', 'fixed_epsilon', 'output_padding=1', 'Normal.rsample()', 'interpolation_codes',
      'два KL с разными аргументами', 'стандартного отклонения', '0,596574', 'одна схема',
      'NIPS 2015 workshop', 'мягкие цели', 'среднему восстановлению']) expect(text).toContain(phrase);
    for (const id of ['почему-обычный-latent-space-неудобен-для-генерации',
      'elbo-reconstruction-и-regularization-latent-space', 'краткие-итоги']) expect(html).toContain(`id="${id}"`);
    for (const name of ['uva-c7d9e18f-MIT.txt', 'pytorch-examples-acc295dc-BSD-3-Clause.txt',
      'conv-arithmetic-af6f818b-MIT.txt']) {
      expect(html).toContain(name);
      await access(join(distDir, 'licenses', name));
    }
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('Это исходный пример прежней редакции');
    expect(html).toContain('f_\\theta(g_\\phi(x_i))');
    expect(html).toContain('/textbook/neural-networks/cnn-lenet-resnet/');
    expect(html).toContain('/textbook/text-representation/numerical-representation/');
  });
  it('renders CNN mechanisms, twenty-one executable examples and fifteen original figures', async () => {
    const html = await readFile(join(distDir, 'textbook/neural-networks/cnn-lenet-resnet/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(21);
    for (const name of ['d2l-cross-correlation.svg', 'cs231n-local-connectivity.jpeg',
      'conv-multi-in.svg', 'conv-1x1.svg', 'cs231n-stride-padding.jpeg', 'pooling.svg',
      'lenet.svg', 'vgg.svg', 'd2l-resnet-block.svg', 'resnet18.svg',
      'escnn-feature-fields.png', 'escnn-vectorfield.gif', 'escnn-conventional-cnn.gif',
      'harvard-window-rows.svg', 'dolz-workflow.png']) expect(html).toContain(name);
    for (const phrase of ['4,9152', '61706', 'F.unfold', 'make_resnet18',
      'zero_gradient', 'не означает ошибку', 'не гарантирует', 'Height и width',
      'код AlexNet в D2L задаёт', 'реальных пикселей в ней меньше']) expect(text).toContain(phrase);
    for (const id of ['двумерная-cross-correlation', 'padding-stride-и-размер-выхода',
      'receptive-field', 'lenet-5', 'alexnet-vgg-и-переход-к-глубине', 'resnet',
      'связь-cnn-с-текстом-и-современными-моделями', 'краткие-итоги']) expect(html).toContain(`id="${id}"`);
    for (const phrase of ['lift_c4', 'roll', 'Rotation-equivariant feature maps',
      'Copyright and source license', 'NO EXPRESS OR IMPLIED LICENSES',
      'Qualcomm Innovation Center']) expect(text).toContain(phrase);
    for (const id of ['rotation-equivariant-feature-maps',
      'four-orientations-from-one-learned-filter', 'read-the-original-steerable-cnn-demonstration'])
      expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/neural-networks/dropout-regularization/"');
    expect(html).toContain('href="/textbook/neural-networks/autoencoder-vae/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders dropout derivations, eleven runnable examples and five full source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/neural-networks/dropout-regularization/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(11);
    for (const name of ['dropout.svg', 'cs231n-dropout-train.jpeg', 'data_augmentation.png', 'd2l-training.svg', 'dropconnect-original.png'])
      expect(html).toContain(name);
    for (const phrase of ['all_masks', 'linear_predictions', 'grad=None', 'state_dict()',
      'restore_best_weights', 'val_acc', 'checkpoint_model', 'training=self.training']) expect(text).toContain(phrase);
    for (const id of ['inverted-dropout', 'один-прямой-проход-в-числах', 'weight-decay',
      'data-augmentation', 'early-stopping', 'краткие-итоги', 'dropconnect', 'stochastic-depth']) expect(html).toContain(`id="${id}"`);
    expect(html).toContain('/licenses/timm-beabdc3a-Apache-2.0.txt');
    expect(html).not.toContain('katex-error');
  });
  it('renders evaluation calculations, complete request accounting and three source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/recommender-systems/evaluation-operations/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(7);
    for (const figure of ['precision-recall.svg', 'diversity.svg', 'linux-foundation-recommenders-architecture.png'])
      expect(html).toContain(figure);
    for (const phrase of ['AP учитывает точность', 'macro average', 'пустой набор', 'col_relevance',
      'recommendForAllUsers(10)', 'intention-to-treat', 'совместимый набор']) expect(text).toContain(phrase);
    for (const id of ['rating-metrics', 'ranking-metrics', 'candidate-set--часть-результата',
      'beyond-accuracy', 're-ranking-как-constrained-optimization', 'production-architecture', 'ab-test'])
      expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/recommender-systems/sequential-feature-aware/"');
    expect(html).toContain('href="/textbook/neural-networks/mlp/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders sequential windows, causal masks, SSE distinctions and eleven executable examples', async () => {
    const html = await readFile(join(distDir, 'textbook/recommender-systems/sequential-feature-aware/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(11);
    for (const name of ['rec-seq-data.svg', 'rec-caser.svg', 'cell-001-attachment.png']) expect(html).toContain(name);
    for (const phrase of ['третье полное окно', 'SSE-PT', 'такой случайной замены embeddings нет',
      'сумму координат', 'Нет настоящих targets', 'available_at', 'pad_left']) expect(text).toContain(phrase);
    for (const id of ['как-матрица-теряет-задачу', 'сначала-split-потом-окна', 'caser-свёртки-по-истории',
      'sasrec-causal-self-attention', 'padding-и-mask--разные-вещи', 'long-term-short-term-и-context',
      'feature-rich-ctr-prediction', 'worked-example-field-interactions', 'признаки-без-утечки',
      'cold-start', 'failure-modes-sequence-models', 'production-contract', 'первоисточники-и-продолжение'])
      expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/recommender-systems/ranking-neural-models/"');
    expect(html).toContain('href="/textbook/recommender-systems/evaluation-operations/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders ranking, DLRM and cold-ID selection with twenty-three examples and seven original figures', async () => {
    const html = await readFile(join(distDir, 'textbook/recommender-systems/ranking-neural-models/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(23);
    expect(html).toContain('id="cold-embedding-ids"');
    expect(html).toContain('torchrec-zch-dataflow.png');
    for (const name of ['rec-ranking.svg', 'rec-neumf.svg', 'rec-deepfm.svg']) expect(html).toContain(name);
    for (const name of ['torchrec-sharding.png', 'torchrec-forward.png',
      'torchrec-full-training-loop.png']) expect(html).toContain(name);
    for (const phrase of ['Which pairwise products reach the top MLP',
      'Backpropagation through pooling and dot products', '476.84 GiB',
      'Incorrect mean of shard means', 'original DLRM paper']) expect(text).toContain(phrase);
    for (const name of ['dlrm-MIT.txt','torchrec-BSD-3-Clause.txt']) {
      expect(html).toContain('href="/licenses/'+name+'"');
      expect((await readFile(join(distDir,'licenses',name),'utf8')).length).toBeGreaterThan(1000);
    }
    for (const phrase of ['попарного loss', '0,3133', 'частотно-взвешенная регуляризация',
      'маску наблюдений', 'натуральный логарифм', 'Суммарная производная logit']) expect(text).toContain(phrase);
    for (const id of ['pointwise-pairwise-и-listwise-постановки', 'bayesian-personalized-ranking',
      'как-loss-реагирует-на-margin', 'bpr-не-превращает-unseen-в-настоящий-dislike',
      'hinge-ranking-loss', 'autorec-восстановление-строки-матрицы', 'neural-collaborative-filtering',
      'формы-тензоров', 'почему-comparison-часто-нечестен', 'factorization-machines-interactions-между-полями',
      'deepfm-low-order-и-high-order-interactions', 'сквозной-training-protocol-для-ranker',
      'failure-modes', 'что-должно-быть-в-отчёте', 'практика-и-первоисточники']) expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/recommender-systems/collaborative-filtering-matrix-factorization/"');
    expect(html).toContain('href="/textbook/recommender-systems/sequential-feature-aware/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders full collaborative-filtering examples, three source figures and the weighted ALS explanation', async () => {
    const html=await readFile(join(distDir,'textbook/recommender-systems/collaborative-filtering-matrix-factorization/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(11);
    for(const name of ['sar-schema.svg','rec-mf.svg','weighted-objectives.svg']) expect(html).toContain(name);
    for(const phrase of ['Weighted ALS','0,40625','вклад','глобальной регуляризацией',
      'всех факторов','матрицы пользовательских']) expect(text).toContain(phrase);
    for(const id of ['матрица-взаимодействий','три-baseline-до-factorization','memory-based-collaborative-filtering',
      'worked-example-sar','matrix-factorization','один-шаг-sgd','implicit-feedback-ноль-состоит-из-разных-причин',
      'от-обучения-к-выдаче','где-модель-ломается','контрольный-эксперимент','первоисточники-и-продолжение'])
      expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/recommender-systems/task-data-pipeline/"');
    expect(html).toContain('href="/textbook/recommender-systems/ranking-neural-models/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders the recommendation-data chapter with seven executable examples and six source figures', async () => {
    const html=await readFile(join(distDir,'textbook/recommender-systems/task-data-pipeline/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(7);
    for(const name of ['google-process.svg','feast-features.png','feast-decisions.png',
      'feast-asof.png','feast-joined.png','rec-ranking.svg']) expect(html).toContain(name);
    for(const value of ['999','watermark','filter_by_created_timestamp','available_at',
      'не зарегистрирован целевой отклик']) {
      expect(text.toLowerCase()).toContain(value.toLowerCase());
    }
    for(const id of ['сначала-определить-событие-и-решение','четыре-разные-задачи',
      'rating-prediction','top-kkk-recommendation','ctr-или-response-prediction',
      'next-item-prediction','двухступенчатый-serving','split-должен-имитировать-будущее',
      'worked-example-временного-split','как-строятся-negatives','candidate-generation--часть-задачи',
      'data-contract','baseline-раньше-архитектуры','bias-baseline-как-проверка-dataset',
      'типичные-ошибки-постановки','практика-и-первоисточники']) expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/recommender-systems/collaborative-filtering-matrix-factorization/"');
    expect(html).toContain('href="/licenses/feast-1e5e90ca-Apache-2.0.txt"');
    expect(html).not.toContain('katex-error');
  });
  it('renders the recommender map with prerequisites, original feedback and process figures', async () => {
    const html=await readFile(join(distDir,'textbook/recommender-systems/index.html'),'utf8');
    for(const name of ['rec-intro.svg','google-process.svg']) expect(html).toContain(name);
    for(const route of ['task-data-pipeline','collaborative-filtering-matrix-factorization',
      'ranking-neural-models','sequential-feature-aware','evaluation-operations'])
      expect(html).toContain(`href="/textbook/recommender-systems/${route}/"`);
    expect(html).toContain('href="/textbook/neural-networks/mlp/"');
    for(const id of ['почему-это-отдельная-дисциплина','три-слоя-системы','минимальный-сквозной-dataset',
      'порядок-чтения','один-запрос-как-маршрут-по-модулю','сквозной-пример','один-лог--разные-targets',
      'два-маршрута-чтения','что-должно-остаться-после-модуля','практика-и-первоисточники'])
      expect(html).toContain(`id="${id}"`);
    expect(html).not.toContain('katex-error');
  });
  it('renders kernel and SVM derivations, nine executable examples and eight source figures', async () => {
    const html=await readFile(join(distDir,'textbook/classical-ml/kernels-svm-rff/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(9);
    for(const name of ['jake-separators.png','jake-support-vectors.png','jake-support-invariance.png',
      'jake-circles-linear.png','jake-radial-lift.png','jake-circles-rbf.png',
      'jake-soft-margin.png','kernel-approximation.png']) expect(html).toContain(name);
    for(const phrase of ['Условия KKT','0,0728','псевдообратную',
      'CC BY-NC-ND 3.0 Unported','Как число признаков влияет на ошибку'])
      expect(text).toContain(phrase);
    expect(text.toLowerCase()).toContain('при равенстве');
    for(const id of ['kernel-как-скалярное-произведение','полиномиальный-kernel-без-магии',
      'svm-максимальный-зазор','линейно-разделимый-случай','soft-margin-и-hinge-loss',
      'от-dual-problem-к-kernel-svm','от-объектов-к-gram-matrix',
      'что-означает-kernel-trick-вычислительно','random-fourier-features-явное-приближение-rbf',
      'численный-пример-одного-приближения','алгоритм-rff-pipeline',
      'nyström-другой-путь-к-явным-признакам','практический-выбор',
      'диагностика-и-типичные-ошибки','leakage-и-scaling','совместная-роль-ccc-и-γgammaγ',
      'accuracy-недостаточна','бюджет-inference','случайность-rff','практика-и-первоисточники'])
      expect(html).toContain(`id="${id}"`);
    expect(html).toContain('href="/textbook/classical-ml/gmm-em/"');
    expect(html).toContain('href="/textbook/recommender-systems/task-data-pipeline/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders the Gaussian-mixture derivations, eight executable fences and seven original figures', async () => {
    const html=await readFile(join(distDir,'textbook/classical-ml/gmm-em/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(8);
    for(const name of ['jake-uncertainty.png','jake-covariance-types.png','jake-stretched.png',
      'jake-moons-two.png','jake-moons-sixteen.png','jake-aic-bic.png','gmm-covariances.png'])
      expect(html).toContain(name);
    for(const value of ['Уверенная принадлежность','суммарное обучающее правдоподобие',
      'Stanford CS228','231 новый параметр','CC BY-NC-ND 3.0 Unported','явное ограничение на ковариацию'])
      expect(text).toContain(value);
    expect(html).toContain('id="почему-likelihood-не-уменьшается"');
    expect(html).toContain('href="/textbook/classical-ml/dimensionality-reduction/"');
    expect(html).toContain('href="/textbook/classical-ml/kernels-svm-rff/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders the full PCA/t-SNE/UMAP chapter, seven executable fences and nine original figures', async () => {
    const html=await readFile(join(distDir,'textbook/classical-ml/dimensionality-reduction/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(7);
    for(const name of ['jake-pca-projection.png','jake-pca-whitening.png',
      'distill-cluster-sizes-input.png','distill-cluster-sizes-tsne30.png',
      'umap-open-cover.png','umap-adaptive-cover.png','umap-fuzzy-graph.png',
      'pca-iris.png','tsne-perplexity.png']) expect(html).toContain(name);
    for(const value of ['Whitening меняет масштаб','оценка градиента смещена',
      'trustworthiness','CC BY-NC-ND 3.0 Unported','5000 итераций','исходных признаков Iris'])
      expect(text).toContain(value);
    expect(html).toContain('id="где-pca-обманывает"');
    expect(html).toContain('id="сравнение-без-ложных-обещаний"');
    expect(html).toContain('href="/textbook/classical-ml/spectral-clustering/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders the spectral-clustering derivation, original figures and four actual code examples', async () => {
    const html=await readFile(join(distDir,'textbook/classical-ml/spectral-clustering/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(4);
    for(const name of ['wiskott-heat-components.png','sklearn-segmentation-four.png',
      'sklearn-segmentation-two.png','spectral-clustering-comparison.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const value of ['Normalized Cut','Ng–Jordan–Weiss','Shi–Malik',
      '1.9.1','0,516047','теплоёмкостью','стационарного распределения',
      'Fabian Schönfeld','CC BY-SA 4.0']) expect(text).toContain(value);
    for(const route of ['clustering','dimensionality-reduction'])
      expect(html).toContain('href="/textbook/classical-ml/'+route+'/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders all four original Lloyd steps and the normalized-center counterexample', async () => {
    const html = await readFile(join(distDir,'textbook/classical-ml/clustering/index.html'),'utf8');
    const text = html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(3);
    for (const step of [1,2,3,4])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/lloyd-step-'+step+'\\.svg"'));
    for (const value of ['Weston.pace','CC BY-SA 3.0','0,64','0,4',
      'Dhillon','Нормировка среднего','не могут взаимно уничтожиться',
      'mutual','AMI','approximate_predict']) expect(text).toContain(value);
    expect(html).toContain('href="/textbook/classical-ml/spectral-clustering/"');
    expect(html).not.toContain('katex-error');
  });
  it('connects both scheduling editions to the interactive allocation exercise', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale + 'textbook/inference/scheduling/index.html'), 'utf8');
      expect(html).toContain('href="/' + locale + 'practice/inference-scheduler/"');
      expect(html).toContain('Interactive scheduler and KV-memory lab');
      expect(html).toContain('does not run a neural network or predict GPU latency');
    }
  });
  it('renders a working canonical-language scheduler in EN instead of redirecting to RU', async () => {
    const html = await readFile(join(distDir, 'en/practice/inference-scheduler/index.html'), 'utf8');
    expect(html).not.toContain('http-equiv="refresh"');
    expect(html).toContain('data-scheduler-lab');
    expect(html).toContain('id="scheduler-step"');
    expect(html).toContain('Canonical-language exercise.');
    expect(html).toContain('href="/en/textbook/inference/scheduling/"');
  });
  it('renders the mixed scheduling walkthrough, two complete HF originals and six executable fences', async () => {
    const html = await readFile(join(distDir,'textbook/inference/scheduling/index.html'),'utf8');
    const text = html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
    for (const name of ['dynamic','continuous'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/scheduling-2026-10-25/hf-'+name+'-batching\\.png"'));
    for (const name of ['anyscale-static-batching.png','gordic-chunked-prefill.png',
      'vllm-prefix-cache-overview.png','vllm-prefix-release-shared-blocks.png','sglang-radix-tree-original.jpg'])
      expect(html).toContain(name);
    for (const value of ['packed_causal_mask','key_start','CC BY-NC-SA 4.0',
      'Rémi Ouazan Reboul','пятнадцати столбцов','num_computed_tokens'])
      expect(text).toContain(value);
    expect(html).toContain('href="/textbook/inference/serving-engines/#'+
      encodeURIComponent('проверка-упакованного-внимания-на-cpu')+'"');
    expect(html).not.toContain('katex-error');
  });
  it('preserves five actual KV-cache examples, complete source diagrams and adjacent routes', async () => {
    const html = await readFile(join(distDir,'textbook/inference/kv-cache-batching-pagedattention/index.html'),'utf8');
    const text = html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(5);
    expect(html).toMatch(/<img[^>]+src="[^"]*attention-review-2026-09\/prefill-decode\.png"/);
    expect(html).not.toMatch(/<img[^>]+src="[^"]*topics-53-59-source-first\/prefill-decode\.png"/);
    for (const name of ['paged-attention-fragmentation','paged-attention-blocks',
      'paged-attention-logical','paged-attention-sharing','paged-attention-copy-on-write'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name+'\\.png"'));
    for (const value of ['Baolin Li','DynamicCache','cache_position','position_ids',
      'cached_logits','generation_cache','padding_cache','131072','copy-on-write'])
      expect(text).toContain(value);
    expect(html).toContain('href="/textbook/inference/llm-inference-physics/"');
    expect(html).toContain('href="/textbook/inference/scheduling/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders six executable decoding examples, source illustrations and stop/beam limits', async () => {
    const html = await readFile(join(distDir,'textbook/inference/decoding/index.html'),'utf8');
    const text = html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
    for (const name of ['greedy_search','beam_search','sampling_search',
      'sampling_search_with_temp','top_k_sampling','top_p_sampling'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/hf-'+name+'\\.png"'));
    for (const value of ['probabilities_at','k_then_p','p_then_k','prompt_ignore_length',
      'StopStringCriteria','beam_indices','compute_transition_scores',
      '0,099','0,02','0,68','0,99','0,0816']) {
      expect(text).toContain(value);
    }
    expect(text).not.toContain('Top-k на реальном распределении GPT-2');
    expect(text).not.toContain('chapter 8');
    expect(text).toContain('speculative sampling и Triton');
    expect(html).toContain('href="/textbook/pretraining/synthetic-data-curricula/"');
    expect(html).toContain('href="/textbook/inference/llm-inference-physics/"');
    expect(html).not.toContain('katex-error');
  });
  it('preserves synthetic-data examples, original figures and the linear chapter route', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/synthetic-data-curricula/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(2);
    expect(html.match(/<pre data-language="bash"/g)).toHaveLength(1);
    for (const value of ['b-cat', '970 млрд', '10 млн отобранного', '1,6 трлн',
      'Self-Instruct', 'OpenThoughts', 'SWE-Rebench', 'Gerstgrasser'])
      expect(text).toContain(value);
    for (const name of ['nvidia-generate-critique-filter.png', 'self-instruct-pipeline.png',
      'nemotron-transformations.png', 'openthoughts-pipeline.png', 'swe-smith.png',
      'swe-rebench.png', 'swezero-noexec.png', 'olmo2-dolmino.png', 'shumailov-collapse.png',
      'phi1-data-ablation.png', 'phi4-repetitions.png', 'nemotron4-prompts.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const anchor of ['synthetic-transformations', 'teacher-data',
      'posttraining-environments', 'curricula-replay', 'synthetic-evidence',
      'phi-synthetic-recipes', 'nemotron4-synthetic-alignment'])
      expect(html).toContain('id="' + anchor + '"');
    expect(html).toContain('href="/textbook/reasoning/test-time-compute/"');
    expect(html).toContain('href="/textbook/inference/decoding/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders RLVR reward gradients, prompt/grader contracts and complete Tulu evidence', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/rlvr-verifiers/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
    for (const value of ['ascent_gradient', 'updated_reward', 'log_failure', 'include_stop_str_in_output',
      '0.13125', '0.488', 'allow_set_relation_comp', 'градиентный подъём',
      'single_tool_reward', 'completion_efficiency_reward', 'greedily consumes']) expect(text).toContain(value);
    expect(html).toContain('id="agent-reward-design"');
    expect(html).toContain('/licenses/toolrl-8cee13ec-Apache-2.0.txt');
    expect(html).toMatch(/<img[^>]+src="[^"]*toolrl\/reward-design\.png"/);
    for (const name of ['tulu-rlvr-loop.png', 'tulu-rlvr-full-figure19.png',
      'tulu-ifeval-kl-and-generalization.png', 'tulu-value-initialization.png', 'tulu-rlvr-reward-mixing.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).not.toMatch(/<img[^>]+src="[^"]*tulu-rlvr-training-curves\.png/);
    for (const id of ['rlvr-contract', 'verifiable-reward', 'parser-vs-verifier', 'grader-contract', 'verifier-audit', 'rlvr-failures', 'главное-различие'])
      expect(html).toContain(`id="${id}"`);
    expect(html).not.toContain('katex-error');
  });
  it('renders actual DPO training, iterative pairing and complete source Tulu figures', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/dpo/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['optimal_probability', 'recovered_reward', 'model_response_logp',
      'pair_attention', 'reference_pair_logp', 'updated_loss', 'reasoning_pairs',
      '0.713015', '9/13', '0.425557', '120 тысяч']) expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
    for (const name of ['fig1.png', 'reward-kl-frontier.png', 'm02-p74-irpo-loop.png',
      'tulu-preference-pipeline.png', 'tulu-prompt-policy-ablations.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const anchor of ['dpo-rhf-objective', 'dpo-derivation', 'dpo-dataflow', 'dpo-offline-boundary', 'dpo-failures', 'проверка-понимания'])
      expect(html).toContain(`id="${anchor}"`);
    expect(html).not.toContain('katex-error');
  });
  it('renders all six executable PPO fences, four original figures and stable mechanism anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/ppo/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['baseline_results', 'ppo_targets', 'ppo_clip_loss',
      '0.499425', 'expected_gradient', 'response_forward', 'sampled_logp',
      'clip_grad_norm_', '0.0475'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
    for (const name of ['value-fn-training.png', 'rlhf-overview.png', 'ppo-clip-viz.png', 'ppo-framework.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const anchor of ['reinforce', 'baseline-advantage', 'ppo-roles', 'ppo-clip', 'ppo-reward-shaping', 'ppo-stale-rollouts'])
      expect(html).toContain(`id="${anchor}"`);
    expect(html).not.toContain('katex-error');
  });
  it('renders runnable reward modeling, listwise/step examples and the complete calibration figure', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/reward-modeling/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['LlamaModel', 'position_ids', 'left_score', 'plackett_luce_loss',
      '0.380952', '0.355556', 'prm_product', 'calibration_loss', '0.929074'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(6);
    for (const name of ['reward-model-hf.png', 'preference-rm-training.png',
      'prm-step-supervision.png', 'self-rewarding-loop.png', 'm02-p89-meta-rewarding-loop.png',
      'evalplanner-plan-execution-loop.png', 'reward-overoptimization-best-of-n.png', 'hh-calibration.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).toContain('id="bradley-terry"');
    expect(html).toContain('id="reward-overoptimization"');
    expect(html).not.toContain('katex-error');
  });
  it('renders executable preference preparation with the complete licensed primary figures', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/preference-data/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['final_turn_pair', 'normalize_verdict', 'ranking_to_pairs',
      'rank_groups', 'validation_ids', 'Chosen', 'Rejected',
      'Do you know why turkeys became the official food of thanksgiving?'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(5);
    expect(html.match(/<pre data-language="json"/g)).toHaveLength(1);
    for (const name of ['hh-workflow.png', 'hh-feedback-interface.png', 'hh-data-distributions.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).not.toMatch(/<img[^>]+src="[^"]*(instructgpt-labeling-ui-002|preference-collection-interface)\.png/);
    expect(html).toContain('id="ранжирование-kkk-ответов"');
    expect(text).toContain('cycle_probability');
    expect(text).toContain('9,5%');
    expect(text).toContain('три вероятности');
    expect(html).toContain('id="задания"');
    expect(html).not.toContain('katex-error');
  });
  it('renders executable SFT loss/update/packing and preserves the six course illustrations', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/sft-instruction-data/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['assistant_only_loss', 'LlamaConfig', 'padded_assistant',
      'block_mask', 'ordinary_changed', '4.840772', '900 000', '§4.2.1'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(3);
    for (const name of ['chatgpt-training-pipeline.png', 'sft-chat-template.png',
      'self-instruct-pipeline.png', 'm04-p30-training-data-gates.png',
      'm04-p54-voting-filter.png', 'minicpm-two-stage-mixture.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(text).not.toContain('В дополнительной части Assignment 5 упаковка должна сохранять EOS');
    expect(html).not.toContain('katex-error');
  });
  it('renders the original Tulu overview without treating the methods as one compulsory chain', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/module-map/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toMatch(/<img[^>]+src="[^"]*\/tulu3-overview\.png"/);
    expect(text).toContain('не образуют обязательную последовательность');
    expect(text).toContain('Unseen evals');
    expect(html).toContain('id="tülu-как-сквозной-открытый-recipe"');
    expect(text).not.toContain('полезен здесь не как ещё одно имя');
    expect(html).not.toContain('katex-error');
  });
  it('keeps the Constitutional AI prerequisite route and complete worked examples in both interfaces', async () => {
    for (const locale of ['', 'en/']) {
      const cases = [
        ['module-map', 'next', 'sft-instruction-data'],
        ['sft-instruction-data', 'prev', 'module-map'],
        ['ppo', 'next', 'alignment-goals'],
        ['alignment-goals', 'prev', 'ppo'],
        ['alignment-goals', 'next', 'dpo'],
        ['dpo', 'prev', 'alignment-goals'],
      ];
      for (const [source, rel, target] of cases) {
        const html = await readFile(join(distDir, locale,
          'textbook/post-training', source, 'index.html'), 'utf8');
        const button = (html.match(/<a\b[^>]*>/g) ?? [])
          .find(tag => tag.includes('rel="' + rel + '"'));
        expect(button, locale + source + ': ' + rel)
          .toContain('href="/' + locale + 'textbook/post-training/' + target + '/"');
      }
      const html = await readFile(join(distDir, locale,
        'textbook/post-training/alignment-goals/index.html'), 'utf8');
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(2);
      for (const image of ['constitutional-ai-overview.png', 'revisions-tradeoff.png'])
        expect(html).toMatch(new RegExp('<img[^>]+src="[^\"]*/' + image.replaceAll('.', '\\.') + '"'));
      expect(html).toContain('href="/' + locale + 'textbook/post-training/reward-modeling/#bradley-terry"');
      expect(html).not.toContain('can be read before SFT');
      expect(html).not.toContain('katex-error');
      if (locale) expectCanonicalEditionNotice(html);
    }
  });
  it('renders the full-group restart example, straggler calculation and original scheduling plots', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/fault-tolerance-orchestration/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['run_recovery', 'pack_state', 'INJECTED_EXIT', 'TORCH_DISABLE_SHARE_RDZV_TCP_STORE',
      'LOADED', 'VERIFIED', '900', '19 из 64 GPU'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(2);
    for (const name of ['harvard-checkpoint-tradeoff.png', 'harvard-cluster-fragmentation.png',
      'failure-domains.svg', 'failure-types.svg', 'gang-deadlock.svg', 'topology-placement.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).not.toMatch(/<img[^>]+src="[^"]*(young-daly-optimization|checkpoint-recovery-timeline)\.svg/);
    expect(html).not.toContain('katex-error');
  });
  it('renders executable DCP/DTensor/FSDP restoration and distinguishes DeepSpeed UCP', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/network-storage-checkpoints/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['reshard_vector', 'verify_resume', 'prepare_fsdp_training',
      'DeepSpeed Universal Checkpointing', '716.1 TB'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(8);
    for (const name of ['jax-superpod-diagram.png', 'deepspeed-ucp-flowchart.png',
      'openhackathons-gpudirect-rdma.png', 'pytorch-async-checkpoint-flow.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).not.toMatch(/<img[^>]+src="[^"]*five-level-model\.svg/);
    expect(html).not.toContain('katex-error');
  });
  it('renders actual EP dispatch code, FSDP motivation and the full source EP illustration', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/expert-hybrid-parallelism/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['ExchangeRows', 'expert_parallel_linear', 'received_experts', 'grad_rows',
      '31.5 MiB', '21 GB', '57 ms', 'ETP=1, EP=64', '-0.375'])
      expect(text).toContain(value);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(2);
    for (const name of ['hf-ep-layer.svg', 'mcore-token-capacity.png',
      'edls-moe-tensor-parallel.png', 'edls-moe-expert-parallel.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).toContain('id="tp-или-ep"');
    expect(html).not.toContain('katex-error');
  });
  it('renders FSDP ownership, actual hybrid reductions and six complete source illustrations', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/zero-fsdp2-devicemesh-dtensor/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['owned_adam_step', 'correct_relu', 'wrong_relu', 'row_sum_shard',
      'reduce_scatter_single', '29.3125 GB', '15.53125 GB', '7 GB', '5.25 GB'])
      expect(text).toContain(value);
    for (const name of ['hf-zero-memory.png', 'hf-fsdp-forward.png', 'hf-fsdp-backward.png',
      'pytorch-fsdp-figure1-overview.png', 'pytorch-fsdp-figure4-hybrid-groups.png', 'edls-dtensor-layouts.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(8);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="cs336-distributed-fsdp"');
    expect(html).toContain('href="/licenses/ultrascale-Apache-2.0.txt"');
    expect(html).toContain('href="/textbook/pretraining/pipeline-parallelism/"');
  });
  it('renders pipeline autograd and nine original figures including complete Zero Bubble schedules', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/pipeline-parallelism/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['pipeline_step', 'partition_costs', 'saved_x.T @ saved_dy',
      '384 MiB', '896 MiB', '12 GiB', '28 GiB', '4/8/6', 'flex_pp', 'post-validation', 'BatchNorm'])
      expect(text).toContain(value);
    for (const name of ['hf-naive-pipeline.svg', 'hf-pipeline-gpipe.svg',
      'hf-pipeline-1f1b.svg', 'hf-interleaved-1f1b.svg', 'dependency.png',
      'zb1p.png', 'zb2p.png', 'zbv.png', 'optimizer.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(7);
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(3);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="cs336-distributed-pipeline"');
    expect(html).toContain('href="/licenses/zero-bubble-Apache-2.0.txt"');
    expect(text).toContain('24 такта работы');
    expect(text).toContain('семь наборов активаций');
    expect(html).toContain('href="/textbook/pretraining/tensor-sequence-parallelism/"');
    expect(html).toContain('href="/textbook/pretraining/zero-fsdp2-devicemesh-dtensor/"');
  });
  it('renders vocabulary-parallel loss, TP arithmetic, FSDP shapes and animated context-parallel sources', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/tensor-sequence-parallelism/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['vocab_cross_entropy', '0.440190', 'mlp_backward',
      '179.2e12', '4.027 ms', '2.600 ms', 'loss_parallel', '84', '28 MiB', 'FSDP×TP'])
      expect(text).toContain(value);
    for (const name of ['tensor-parallel-split.svg', 'hf-attention-tp.png', 'hf-tp-sp-regions.png',
      'hf-causal-zigzag.svg', 'ulysses-design.png', 'hf-ring-attention.gif'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(3);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/licenses/ultrascale-Apache-2.0.txt"');
    expect(html).toContain('href="/textbook/pretraining/gradient-checkpointing-offload/"');
    expect(html).toContain('href="/textbook/pretraining/pipeline-parallelism/"');
  });
  it('renders checkpoint storage aliasing, budget selection, originals and complete code', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/gradient-checkpointing-offload/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['view = base[:1]', '4 194 304', 'choose_saved',
      'torch._functorch.config', '7 ms', '1.4317 с', '2.75', '3.75'])
      expect(text).toContain(value);
    for (const name of ['pytorch-eager-saved-tensors.png', 'pytorch-memory-budget-pareto.png',
      'pytorch-checkpoint.png', 'pytorch-selective-checkpoint.png', 'deepspeed-optimizer-offload.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(8);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/pretraining/processes-collectives-ddp/"');
    expect(html).toContain('href="/textbook/pretraining/tensor-sequence-parallelism/"');
  });
  it('renders DDP flattening, explicit ring rounds, DP capacity and source compression diagrams', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/processes-collectives-ddp/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['flat = torch.cat', '[2,4,6,8,10,12]',
      'alternate_ring_all_reduce', '5,285 ms', '4,624 ms', '5,395 ms',
      'до 13 ms', 'до 11 ms', 'Algorithm 1', 'divide_by_initial_world_size'])
      expect(text).toContain(value);
    for (const name of ['hf-ddp-sequential.svg', 'hf-ddp-overlap.svg', 'edls-error-feedback.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(15);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/licenses/edls-e632aa89-MIT.txt"');
    expect(html).toContain('href="/textbook/pretraining/gradient-checkpointing-offload/"');
  });
  it('renders scaling-law fitting steps, original ablation plots and task calibration', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/scaling-laws/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['three_point_minimum', 'N=6.236B, D=80.185B', 'L-BFGS',
      'LSE', 'However, this led to poor results at smaller batch sizes',
      'the attention logits in a Transformer', 'MuAdam', 'ARC Challenge',
      'липшицева', 'n^{-2/(d+2)}'])
      expect(text).toContain(value);
    for (const name of ['porian-figure1.svg', 'sp-coordinates.png', 'mup-coordinates.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(2);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/licenses/mup-19814971-MIT.txt"');
    expect(html).toContain('href="/textbook/pretraining/next-token-objective/"');
    expect(html).toContain('href="/textbook/pretraining/distributed-systems/"');
  });
  it('renders the distributed overview with visible TP gradients, overlap and published group configurations', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/distributed-systems/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['split_loss.backward()', '[[60.0, 12.0]]',
      '[9.0, 33.0] 42.0 2.0', 'ddp.no_sync()', '4–7', '12–15 ms',
      'fsdp_tp_calcs', '49 устройств', '16 устройств',
      'TP=8, CP=16, PP=16, DP=8', '2048 H800']) expect(text).toContain(value);
    for (const name of ['hf-ddp-sequential.svg', 'hf-ddp-overlap.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^\"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(3);
    expect(html).toContain('device-placement-dimensions.svg');
    for (const fact of ['placement_groups', '[0, 8, 16, 24]', '[7, 15, 23, 31]',
      '32 devices, not eight', 'wrong batch rule']) expect(text).toContain(fact);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/licenses/ultrascale-Apache-2.0.txt"');
    expect(html).toContain('href="/textbook/pretraining/scaling-laws/"');
  });
  it('renders the complete placement SVG, source caption and group example in both locale routes',async()=>{
    for(const prefix of ['', 'en/']){
      const html=await readFile(join(distDir,prefix+'textbook/pretraining/distributed-systems/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      expect(html).toContain('id="three-coordinates-for-one-gpu"');
      expect(html).toContain('device-placement-dimensions.svg');
      for(const fact of ['Device Placement Dimensions','placement_groups','32 devices, not eight',
        '30 sequences','480','not a network topology or pipeline schedule']) expect(text).toContain(fact);
      expect(html).not.toContain('katex-error');
      if(prefix) expectCanonicalEditionNotice(html);
    }
  });

  it('renders finite-context PPL examples, original HF animations and sequential MTP explanation', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/next-token-objective/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['scoring_windows', 'num_valid_tokens - batch_size', '3, 1 и 0',
      '3, 2 и 1', '3{,}1748', '2{,}3784', 'RMSNorm', 'матрица', '16/9'])
      expect(text).toContain(value);
    for (const name of ['d2l-language-model-data.png', 'd2l-gpt-decoder.png', 'mtp-overview.png',
      'ppl-full.gif', 'ppl-chunked.gif', 'ppl-sliding.gif'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(3);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/licenses/transformers-e7b03417-Apache-2.0.txt"');
    expect(html).toContain('href="/textbook/pretraining/scaling-laws/"');
    expect(html).toContain('href="/textbook/pretraining/dedup-pii-corpus-quality/"');
  });
  it('renders text and image deduplication with five original figures and five Python examples', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/dedup-pii-corpus-quality/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['banana', 'longest common prefix', 'Results are based on minhash segment collision',
      '450 полос по 20 значений', 'EditDistance', '0,9946', '11,2%', 'print(verified)']) expect(text).toContain(value);
    for (const name of ['presidio-analyze-anonymize.png', 'lsh-characteristics.png',
      'lee-dedup-cluster-sizes-figure1.png', 'lee-dedup-perplexity-figure2.png', 'semdedup-intro.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre data-language=/g)).toHaveLength(6); // Five Python examples and the JSON audit record.
    for (const value of ['difference_bits', 'perceptual_bits', 'remove_any_earlier',
      'Colour collision', 'caption diversity']) expect(text).toContain(value);
    expect(html).toContain('href="/licenses/imagehash-BSD-2-Clause.txt"');
    expect(html).toContain('href="/licenses/semdedup-CC-BY-NC-4.0.txt"');
    for (const id of ['pii', 'harm-policy', 'exact-dedup', 'minhash-lsh', 'image-deduplication', 'contamination', 'audit-log'])
      expect(html).toContain('id="' + id + '"');
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/pretraining/data-pipeline/"');
    expect(html).toContain('href="/textbook/pretraining/next-token-objective/"');
  });
  it('renders Harvard lab strings with intact TeX and separates runtime templates from original explanations', async () => {
    const records = JSON.parse(await readFile(join(rootDir,
      '05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json'), 'utf8'));
    const labs = records.filter((row: any) => row.kind === 'lab');
    expect(labs).toHaveLength(34);
    for (const row of labs) {
      const html = await readFile(join(distDir, row.route, 'index.html'), 'utf8');
      expect(html, row.route).not.toContain('class="katex-error');
      expect(html).toContain('Static text, not a running lab');
      expect(html).toContain('Complete original Marimo source');
      expect(html).toContain('/45ecc8d82fcae70c149cdce550d3b3d3411df913/LICENSE.md');
      const math = [...html.matchAll(/<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g)]
        .map(match => match[1]);
      expect(math.every((value: string) => !value.includes('`…`')), row.route).toBe(true);
    }
    const edge = await readFile(join(distDir,
      'sources/courses/harvard-ml-systems/labs/vol2/lab-11-edge-intelligence/index.html'), 'utf8');
    expect(edge).toContain('M_{\\text{train}} = M_{\\text{weights}}');
    expect(edge).toContain('UI text and runtime templates (not executed)');
    expect(edge).toContain('<details>');
    const chapter = await readFile(join(distDir, 'textbook/deployment/edge-federated/index.html'), 'utf8');
    expect(chapter).toContain('id="harvard-mixed-state-memory"');
    expect(chapter).toContain('120 MB');
    expect(chapter).toContain('152 MB');
    expect(chapter).toContain('192 MB');
    expect(chapter).toContain('issue in the source');
    expect(chapter).not.toContain('class="katex-error');
  });
  it('publishes finite-product examples with original JPEGs, English source prose and convergence conditions', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/formal-math/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="finite-products-and-missing-assumptions"');
    for (const name of ['prime-factorization.jpg', 'prime-geometric-factors.jpg'])
      expect(html.match(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"', 'g'))).toHaveLength(1);
    for (const value of ['remainders == [17, 17, 100]', 'F(91, 36)',
      'An intermediate step is', 'norm-summable', 'finite partial sums and bounds',
      'non-integer', 'symmetric limit', 'do not establish those theorems']) expect(text).toContain(value);
    expect(html).toContain('href="/sources/papers/mathlib-euler-products/"');
    expect(html).not.toContain('katex-error');
    const card = await readFile(join(distDir, 'sources/papers/mathlib-euler-products/index.html'), 'utf8');
    expect(card).toContain('not an LLM benchmark paper');
    for (const notice of ['formal-zeta-NOTICE.txt', 'mathlib-c44e0c8-Apache-2.0.txt'])
      expect(await readFile(join(distDir, 'licenses', notice), 'utf8')).toBe(
        await readFile(join(rootDir, 'site/public/licenses', notice), 'utf8'));
  });
  it('publishes the OLMo recipe, complete schedule slide and corrected mixture attribution', async () => {
    const html = await readFile(join(distDir, 'textbook/pretraining/data-and-pretraining/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    expect(html).toContain('id="olmo-two-stage-recipe"');
    expect(html.match(/<img[^>]+src="[^"]*m04-p147-two-stage-training\.png"/g)).toHaveLength(1);
    expect(html).toContain('alt="OLMo: warmup, длинная косинусная стадия и короткое линейное снижение до нуля"');
    for (const value of ['truncated normal distribution', 'three separate times for 50B tokens',
      '8,388,608,000', '16.78B', '150B-token sequential continuation',
      '600B tokens', '60.5 to 43.0', 'MMLU-Pro', '43.4 percentage points'])
      expect(text).toContain(value);
    expect(html).toContain('href="https://arxiv.org/html/2501.00656v1#bib.bib123"');
    expect(html).toContain('href="/sources/papers/olmo2-dolmino/"');
    expect(html).not.toContain('katex-error');
    const collection = await readFile(join(distDir, 'textbook/pretraining/data-pipeline/index.html'), 'utf8');
    expect(collection).toContain('William Held');
    expect(collection).not.toContain('Shen и соавторы');
    const card = await readFile(join(distDir, 'sources/papers/olmo2-dolmino/index.html'), 'utf8');
    expect(card.match(/<table[\s>]/g)).toHaveLength(3);
    for (const value of ['21.32T', '2.95B', '832.6B', '739.8B', 'Dolmino SynthMath',
      '35.1M', '7.27K', '21.37M', 'not GPU memory requirements']) expect(card).toContain(value);
    const notice = 'olmo-pretraining-NOTICE.txt';
    expect(await readFile(join(distDir, 'licenses', notice), 'utf8')).toBe(
      await readFile(join(rootDir, 'site/public/licenses', notice), 'utf8'));
  });
  it('renders microanneals, probabilistic filtering and the complete OpenWebMath pipeline', async () => {
    const intro = await readFile(join(distDir, 'textbook/pretraining/data-and-pretraining/index.html'), 'utf8');
    const html = await readFile(join(distDir, 'textbook/pretraining/data-pipeline/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['gpt3_keep_probability', 'Pareto', 'MathScore', '4968',
      '30 000', '15 000', 'contains_math', 'Prefiltering HTML Documents']) expect(text).toContain(value);
    expect(html).toMatch(/<img[^>]+src="[^"]*corpus-collection-2026-10-23\/openwebmath-pipeline\.png"/);
    // The HTML example also contains a literal <pre> inside the copy-button attribute.
    expect(html.match(/<pre data-language=/g)).toHaveLength(4);
    const introText = intro.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['microanneals', '130 млрд', '19 microanneals',
      '200 вопросов', 'Короткие продолжения одного чекпойнта']) expect(introText).toContain(value);
    expect(intro).not.toContain('katex-error');
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/pretraining/dedup-pii-corpus-quality/"');
    expect(intro).toContain('href="/textbook/ml-systems/gpu-kernels-triton/"');
  });
  it('renders the corpus-release distinctions and complete ROOTS figure in both locales', async () => {
    for(const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale+'textbook/pretraining/data-pipeline/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      for(const value of ['20.8 billion documents','30.4 trillion tokens','TB of text',
        'sum to 47%','remove 23.1%','not training-token weights']) expect(text).toContain(value);
      for(const anchor of ['redpajama-releases','roots-multilingual-corpus'])
        expect(html).toContain('id="'+anchor+'"');
      expect(html.match(/<img[^>]+src="[^"]*corpus-releases-2026-10-29\/roots-oscar-filtered\.png"/g))
        .toHaveLength(1);
      expect(html).toContain('aa033f6b6480c445557e9118283993403f069f6a/README.md');
      expect(html).toContain('https://arxiv.org/html/2303.03915v1#S3.F3');
      expect(html).not.toContain('katex-error');
    }
  });
  it('renders Triton derivations, executable chunked CE and nine original figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/gpu-kernels-triton/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['correct_block', 'wrong_block', '0.625', '435982',
      'chunked_linear_ce', 'full_dh', 'full_dw', '125 МиБ', '7.8125 ГиБ',
      '1.357707', '1.723299', 'weight.T']) expect(text).toContain(value);
    for (const name of ['l05-p31.png', 'l05-p32.png', 'l05-p33.png',
      'triton-softmax.png', 'triton-row-sum.png', 'l05-p42.png', 'l05-p43.png',
      'triton-grouped-vs-row-major.png', 'liger-flce-original.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const id of ['cs336-systems-triton', 'контракт-собственного-kernel',
      'что-именно-задаёт-triton-program', 'первый-полный-пример-softmax-одной-строки',
      'reduction-которая-не-помещается-в-один-tile',
      'tiled-gemm-reuse-вместо-идеализированной-загрузки-всей-матрицы',
      'autotuning--поиск-по-корректным-кандидатам', 'как-читать-профиль-triton-kernel'])
      expect(html).toContain('id="' + id + '"');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(45);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/ml-systems/profiling/"');
    expect(html).toContain('href="/textbook/pretraining/data-and-pretraining/"');
  });
  it('renders profiler interpretation, allocator examples and four original screenshots', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/profiling/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['step-5.json', '22 мс', '4001-м', '156 МиБ',
      'record_pinned_host_memory=True', 'host_segments', 'forward_and_loss']) expect(text).toContain(value);
    for (const name of ['pytorch-profiler-trace.png', 'a2-nsight-trace.png',
      'pytorch-active-memory-timeline.png', 'pytorch-allocator-state-history.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const id of ['cs336-systems-profiling', 'полный-диагностический-маршрут',
      'диагноз-по-уровню-стека', 'pytorch-profiler', 'cpu-py-spy', 'nsight-systems',
      'от-python-функции-до-kernel-где-находится-framework', 'capture-не-равен-optimization',
      'interoperability-graph-как-контракт-с-границами', 'memory-snapshot'])
      expect(html).toContain('id="' + id + '"');
    expect(html.match(/<pre/g)).toHaveLength(13);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/ml-systems/data-pipeline-padding-packing/"');
    expect(html).toContain('href="/textbook/ml-systems/gpu-kernels-triton/"');
  });
  it('publishes the full Harvard step and the separate CPU gradient/event walkthrough in both routes',async()=>{
    for(const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale+'textbook/ml-systems/profiling/index.html'),'utf8');
      const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
      for(const fact of ['A complete training step','not a standalone runnable program',
        'observed_step','fixed_input_products','trainable_input_products',
        'expected_gradients','13,008,896','13,172,736','not five CUDA kernel launches'])
        expect(text).toContain(fact);
      expect(html.match(/<pre data-language="python"/g)).toHaveLength(11);
      // The viewer also contains an empty image element without src.
      expect(Array.from(html.matchAll(/<img\b[^>]*\ssrc="([^"]+)"[^>]*>/g))).toHaveLength(5);
      expect(html).toContain('cs231n-two-layer.jpeg');
      expect(html).toContain('href="https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/');
      expect(html).not.toContain('katex-error');
      if(locale) expectCanonicalEditionNotice(html);
    }
  });
  it('renders data-pipeline reasoning, executable packing, caching and six source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/data-pipeline-padding-packing/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['104 мс', '144', '26,5%', '555 806', 'in_order=False',
      'causal_only', 'changed_v', '10, 15, 20, 25', 'Нормировка при усреднении градиентов DDP',
      'содержимое shuffle-буфера теряется', 'Декодирование изображений и DALI',
      '37.5 МиБ', '4.59 МиБ', '18.38 МиБ', 'fn.readers.caffe']) expect(text).toContain(value);
    for (const name of ['pinmem.png', 'same-stream.png', 'separate-stream.png',
      'optimal-sequence-processing.png', 'padding-packing.png', 'cached-dataset.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html.match(/<pre/g)).toHaveLength(9);
    expect(text).toContain('Caching across epochs');
    expect(text).toContain('Persistent cache identity and memory');
    expect(text).toContain('143.55 MiB');
    expect(text).toContain('574.22 MiB');
    expect(html).toContain('id="cache-placement-and-epoch-state"');
    expect(html).toContain('/licenses/tensorflow-docs-e4210b94-Apache-2.0.txt');
    expect(html).not.toContain('katex-error');
    for (const id of ['storage-и-dataloader-как-система', 'packing-без-leakage',
      'базовый-путь-загрузки-без-доказанного-перекрытия', 'проверка-throughput'])
      expect(html).toContain('id="' + id + '"');
    expect(html).toContain('href="/textbook/ml-systems/numerical-formats-mixed-precision/"');
    expect(html).toContain('href="/textbook/ml-systems/profiling/"');
  });
  it('retains the cache examples and full vector in root and explicit English fallback', async () => {
    for(const locale of ['', 'en/']) {
      const html=await readFile(join(distDir,locale+'textbook/ml-systems/data-pipeline-padding-packing/index.html'),'utf8');
      expect(html.match(/<pre\b/g)).toHaveLength(9);
      expect(Array.from(html.matchAll(/<img\b[^>]*\ssrc="([^"]+)"[^>]*>/g))).toHaveLength(6);
      for(const value of ['cache-placement-and-epoch-state','decode_calls','epoch_dependent',
        'reshuffle_each_iteration','records-v1','cached-dataset.svg']) expect(html).toContain(value);
      expect(html).not.toContain('katex-error');
      if(locale) expectCanonicalEditionNotice(html);
    }
  });
  it('renders precision experiments, scaling examples and six source diagrams', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/numerical-formats-mixed-precision/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['repeated_sum', '10.000133514404297', '9.953125',
      '10.00213623046875', 'Current и delayed scaling', '31 единицу', 'reduction="sum"'])
      expect(text).toContain(value);
    for (const name of ['te-float-bits.png', 'hf-format-ranges.png', 'hf-format-resolution.png',
      'te-master-weights.svg', 'te-current-delayed-scaling.svg', 'te-mxfp8-blocks.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="cs336-systems-precision"');
    expect(html).toContain('href="/textbook/ml-systems/transformer-moe-arithmetic/"');
    expect(html).toContain('href="/textbook/ml-systems/data-pipeline-padding-packing/"');
  });
  it('renders transformer arithmetic, the numeric backward and scaling limits', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/transformer-moe-arithmetic/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['finite_difference', '13.0 5.0', 'MFU и HFU', '66.7%',
      '60.0%', '80.0%', 'Сколько устройств', '2.048', '4.128', '1.215', 'общая сеть'])
      expect(text).toContain(value);
    for (const name of ['jax-transformer.png','hf-training-memory.svg','nemo-checkpointing-detail.png',
      'hf-fsdp-forward.svg','hf-fsdp-backward.svg','edls-tp.png','edls-ep.png',
      'hf-pipeline-gpipe.svg','hf-pipeline-1f1b.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="cs336-systems-arithmetic"');
    expect(html).toContain('href="/textbook/ml-systems/performance-measurement-roofline/"');
    expect(html).toContain('href="/textbook/ml-systems/numerical-formats-mixed-precision/"');
  });
  it('renders the performance capstone with source trace, complete examples and evidence limits', async () => {
    const html = await readFile(join(distDir, 'practice/13-optimizirovat-odin-transformer-step/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const value of ['Profile, compare, size, and choose', 'Find the interval that delays the result',
      'Separate an observed difference', 'Plan sample size before seeing the result',
      'Apply constraints before ranking candidates', 'ttest_ind', 'equal_var=False',
      'planned_shift', 'required_units', 'hold: no feasible option', 'not measurements']) expect(text).toContain(value);
    expect(html).toMatch(/<img[^>]+src="[^"]*\/a2-nsight-trace\.png"/);
    expect(html.match(/<pre/g)).toHaveLength(6);
    expect(html).toContain('id="performance-evidence-profiling-regression-capacity"');
    expect(html).toContain('href="/licenses/scipy-1.18.0-LICENSE.txt"');
    expect(html).not.toContain('katex-error');
    expect(text).toContain('TinyTorch');
  });
  it('renders per-level and sequential roofline calculations with nine source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/performance-measurement-roofline/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const name of ['b2b9aaeaa637a9a665f0f4e9229daeb5bc5a4f2d.svg',
      '682e143e450db0d85ad5c608e3a125ac5176e0ba.svg', 'd2l-asyncgraph.svg',
      'd2l-threading.svg', 'jax-roofline.png', 'l05-p47.png', 'a2-nsight-trace.png',
      'megatron-weak-scaling.png', 'megatron-strong-scaling.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const value of ['4.5 TB/s', '90', 'Последовательность операций:',
      'lower_bound_ms', 'sequential_ms', 'pooled_ms', '3.0', '4.8']) expect(text).toContain(value);
    expect(html.match(/<pre/g)).toHaveLength(6);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="cs336-systems-roofline"');
    expect(html).toContain('id="fleet-mfu-scaling-and-preserved-progress"');
    for (const value of ['MFU and HFU count different work', '35446', '26.15%',
      'Preserved progress, checkpointing and recovery', 'Preserved-work utilization:',
      'did not train these models to convergence']) expect(text).toContain(value);
    expect(html).toContain('href="/textbook/ml-systems/gpu-cuda-memory-hierarchy/"');
    expect(html).toContain('href="/textbook/ml-systems/transformer-moe-arithmetic/"');
  });
  it('renders GPU memory examples, fifteen originals and adjacent navigation', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/gpu-cuda-memory-hierarchy/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    const figures = ['cuda-grid-cta.png', 'gpu-hardware.png', 'nvidia-coalescing-4.png',
      'nvidia-coalescing-5.png', 'banks-stride-1.svg', 'banks-stride-2.svg', 'banks-stride-4.svg',
      'nvidia-gemm-step-6.png', 'nvidia-gemm-step-7.png', 'nvidia-gemm-step-8.png',
      'nvidia-gemm-step-9.png', 'triton-grouped-vs-row-major.png', 'nvidia-fusion.svg',
      'nvidia-same-stream.png', 'nvidia-two-streams.png'];
    for (const name of figures)
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const value of ['Swizzling:', 'Неполная плитка:', 'tiled_matmul', '139.', '154.',
      'Tensor Memory', 'sm_100a', '16 KiB']) expect(text).toContain(value);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('id="cs336-systems-gpu"');
    expect(html).toContain('href="/textbook/ml-systems/model-as-part-of-system/"');
    expect(html).toContain('href="/textbook/ml-systems/performance-measurement-roofline/"');
  });
  it('renders end-to-end latency, queueing example and six source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/model-as-part-of-system/index.html'), 'utf8');
    const text = html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ');
    for (const name of ['ml-lifecycle.png', 'mit-amdahl.png', 'mit-pipeline.png',
      '2f10e92057738ef1e4a50e524fddf5389d61d795.svg',
      'c491ee41569a92f002bf915150731598e0d64e26.svg',
      '3b022a3ffa6f028fe2c2c3165c292a16a3a9e0ff.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const s of ['Почему средней скорости недостаточно', '69 мс', '103 мс',
      '35, 54 и 73 мс', '0.94', 'не результат измерения конкретной камеры или GPU'])
      expect(text).toContain(s);
    expect(html).not.toContain('katex-error');
    expect(html).toContain('href="/textbook/modern-llm/sequence-model-alternatives/"');
    expect(html).toContain('href="/textbook/ml-systems/gpu-cuda-memory-hierarchy/"');
  });
  it('renders sequence-model calculations, complete Mamba-2 figures and navigation',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/sequence-model-alternatives/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['mamba2-ssd-algorithm.png','mamba2-block-complete.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/sequence-alternatives-2026-10-23/'+name.replaceAll('.','\\.')+'"'));
    for(const s of ['diagonal_recurrence','selective_ssm','ssd_chunked','normalized_linear',
      'gated_delta','ttt_linear','9.20625','6.28125','17.5'])
      expect(text).toContain(s);
    expect(html).not.toContain('mamba2-block-figure.png');
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/moe\/mixture-of-experts\/"/);
    expect(html).toMatch(/К модулям:[\s\S]{0,300}href="\/textbook\/#/);
  });
  it('renders complete MoE source graphs, worked examples and neighboring navigation',async()=>{
    const html=await readFile(join(distDir,'textbook/moe/mixture-of-experts/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['switch-scaling','switch-capacity','switch-parallelism','olmoe-shared',
      'olmoe-granularity','olmoe-lbl','olmoe-lbltoks','olmoe-zloss'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/moe-2026-10-23/'+name+'\\.png"'));
    for(const s of ['dispatch_combine','capacity_mask','router_losses','node_limited_route',
      '0.99024','route_scale','32 байта'])expect(text).toContain(s);
    expect(html.match(/<pre/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/long-context\/"/);
    expect(html).toMatch(/<strong>Дальше:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/sequence-model-alternatives\/"/);
  });
  it('renders the complete FFN examples, retained animations and original composition plot',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/dense-ffn/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['transformer-block-ffn.png','gpt2-mlp-expand.gif','gpt2-mlp-project.gif',
      'cs336-silu-vs-relu.png','geva-ffn-memory.svg','tensor-parallel-split.svg',
      'mixtral-smoe-layer.png','geva-composition.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['swiglu_backward','fused_projection','split_y','1.633','4000 префиксах',
      '135 266 304','172 MiB'])expect(text).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(3);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/rope\/"/);
    expect(html).toMatch(/<strong>Дальше:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/transformer-shapes-resources\/"/);
  });
  it('renders long-context mechanisms, CSA/HCA originals and adjacent reading links',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/long-context/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['position-interpolation.png','yarn-method-summary.png','mistral-sliding-window.png',
      'mistral-rolling-step-0.png','mistral-rolling-step-1.png','mistral-rolling-step-2.png',
      'cla-diagram.png','deepseek-dsa-architecture.png','lost-in-the-middle.png',
      'mistral-chunk-mask.png','cla-cache-groups.png','ruler-aggregation.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['window_attention','h2_shared','index_scores','13.25','1.375 GiB',
      'пересчитать затронутый префикс','SQuAD и HotpotQA','500 примеров'])expect(text).toContain(fact);
    for (const figure of ['deepseek-v4-csa.png', 'deepseek-v4-hca.png']) expect(html).toContain(figure);
    expect(html).toContain('compressed-sparse-attention-csa-hca');
    expect(html.match(/<pre/g)).toHaveLength(5);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/mla-kv-compression\/"/);
    expect(html).toMatch(/<strong>Дальше:<\/strong>[\s\S]{0,300}href="\/textbook\/moe\/mixture-of-experts\/"/);
  });
  it('renders normalized MLA, positional obstruction and original source panel',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/mla-kv-compression/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['cs336-mha-gqa-mqa-mla.png','mha-mla-projections.svg',
      'deepseek-mla-panel.png','deepseek-v2-mha-gqa-mqa-ablation.png','deepseek-v2-mla-ablation.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const s of ['TinyMLA','same_latents','q_gain','kv_gain','11110','11111',
      '80 чисел','20 чисел','13 254 000 640','8 H800','420B'])expect(text).toContain(s);
    expect(html.match(/<pre/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
    expect(html).not.toMatch(/<img[^>]+src="[^"]*deepseek-v2-figure2-hq/);
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/mha-mqa-gqa\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/long-context\/"/);
  });
  it('renders compact GQA examples, complete original results and adjacent chapter links',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/mha-mqa-gqa/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['prefill-decode.png','cached-first-token.png','next-token-attends-cache.png',
      'mha-gqa-mqa.png','uptrain-recycling.png','task-results.png','conversion-methods.png',
      'uptraining-proportion.png','groups-speed.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['grouped_attention','query_pos','2.5379','3.4621','37.5%',
      '31.88','time per sample per TPUv4 chip','трём запускам'])expect(text).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('/source-first-35-36/gqa-');
    expect(html).toMatch(/К модулям:[\s\S]{0,300}href="\/textbook\/#/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/mla-kv-compression\/"/);
  });
  it('renders complete resource examples, original plots and uncropped optional course pages',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/transformer-shapes-resources/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['a1-figures1-2-model-block.png','compute-memory.png',
      'jax-scaling-roofline.webp','jax-roofline-batch.png',...Array.from({length:10},(_,i)=>'a1-page-'+(i+18)+'.png')])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['parameter_layout','17576448','27795652608','accumulated_gradient',
      '136.1','226.3','20480','27.80%','3.47%'])expect(text).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(6);
    expect(html).not.toContain('katex-error');
    expect(html).not.toContain('/a1-sequence-');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/dense-ffn\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/pretraining\/data-and-pretraining\/"/);
  });
  it('renders RoPE derivation, scaling/cache examples and all five original figures',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/rope/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['roformer-figure1.png','roformer-decay.png','pi-extrapolation.png',
      'position-interpolation.png','yarn-method-summary.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['3.701805','0.00232628','63496.04','1.29648','stale_kv',
      'Причинная маска уже задаёт порядок','не измерение attention logits'])
      expect(text).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
    expect(html).not.toMatch(/<img[^>]+src="[^"]*(rope-implementation|rope-decay)\.png/);
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/norm-gating-residual\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/dense-ffn\/"/);
  });
  it('renders normalization examples, five original figures and the complete stability explanations',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/norm-gating-residual/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['xiong-pre-post-norm.png','xiong-gradient-norms.png',
      'cs336-silu-vs-relu.png','chameleon-norm-qk-dropout.png','chameleon-training-ablations.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['0.03162','1.66667','1.0204','0.1230','4000 шагами warm-up',
      'naive = large','logsumexp','10.0063','порядок округления различается',
      'QK normalization','Z-loss','Logit soft-capping'])expect(text).toContain(fact);
    expect(html).not.toMatch(/<img[^>]+src="[^"]*rmsnorm-paper-results/);
    expect(html.match(/<pre/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/decoder-block\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/rope\/"/);
  });
  it('renders the worked decoder stack, SwiGLU intermediates and all six original figures',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/decoder-block/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['cs336-prenorm-block.png','mha-gqa-mqa.png',
      'gpt2-transformer-block-vectors-2.png','ffn-ratio-evidence-slide.png',
      'aspect-ratio-evidence-slide.png','weight-decay-evidence-slide.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['Два блока и два кэша','Почему кэш для генерации не заменяет граф обучения',
      '−1.5672','1.8936','−0.5123','1.2456','192 байта','copy.deepcopy(model)',
      'строке станет нулевым','0.3164','0.2356','start_pos=1'])expect(text).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(2);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/llama-architecture\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/norm-gating-residual\/"/);
  });
  it('renders LLaMA cached generation, shared-prefix boundaries and seven source figures',async()=>{
    const html=await readFile(join(distDir,'textbook/modern-llm/llama-architecture/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['cs336-model-overview.png','cs336-prenorm-block.png',
      'roformer-figure1.png','cs336-silu-vs-relu.png','llama1-training-loss.png',
      'mha-gqa-mqa.png','llama1-downstream-training.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['def forward(ids, cache=None)', 'query_pos = old + torch.arange(t)',
      'Маска имеет форму 2×5','Последний выбранный ID','2+3+4=9',
      'Когда можно использовать сохранённый префикс','на WinoGrande 33B и 65B',
      '0.068866','6 738 415 616'])expect(text).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(3);
    expect(html).toContain('href="/sources/papers/llama/"');
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/t5\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/decoder-block\/"/);
  });
  it('renders corrected LLaMA paper protocols and the actual original downstream plots',async()=>{
    const html=await readFile(join(distDir,'sources/papers/llama/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const fact of ['27 февраля 2023','52.4','68.5','50.9','69.7',
      'С голосованием по 100','Gender score 70.6','Это не точность WinoGender',
      '0.1 для pass@1','0.8 для pass@100','77.5','4.1 и 5.9'])expect(text).toContain(fact);
    expect(html).toMatch(/<img[^>]+src="[^"]*\/llama1-downstream-training\.png"/);
    expect(html).toContain('href="/textbook/modern-llm/llama-architecture/"');
    expect(text).not.toContain('Gender bias: 70.6% верных');
    expect(html).not.toContain('katex-error');
  });
  it('renders T5 corrected training pairs, cache invalidation and six original figures',async()=>{
    const html=await readFile(join(distDir,'textbook/models/t5/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['t5-figure1.png','t5-figure2-spans.png','t5-figure3.png',
      't5-figure4.png','t5-figure5-objectives.png','t5-figure6-repeated-data.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['[14, 9, 13, 1, -100, -100, -100]',
      'Что можно сохранить между запросами','EncoderDecoderCache.is_updated',
      'сбрасывают и self-KV decoder','Сам график показывает только training loss',
      '59.29 против 71.36','36 и 8 токенов','222 903 552'])expect(text).toContain(fact);
    expect(text).not.toContain('[14, 9, 12, 1, -100');
    expect(html).toContain('href="/sources/papers/t5/"');
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/gpt-3\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/modern-llm\/llama-architecture\/"/);
  });
  it('renders corrected T5 source protocols without fabricated architecture claims',async()=>{
    const html=await readFile(join(distDir,'sources/papers/t5/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const fact of ['23 октября 2019','2023 года','SQuAD 1.1 exact match',
      '91.26','96.22','18 из 24','59.29','32.1','на 24 октября 2019',
      'не выделяет независимый вклад'])expect(text).toContain(fact);
    expect(text).not.toContain('Минимум 3 предложения, 5 слов');
    expect(html).toContain('href="/textbook/models/t5/"');
    expect(html).not.toContain('katex-error');
  });
  it('renders GPT-3 source-faithful protocols, five licensed originals and native configuration data',async()=>{
    const html=await readFile(join(distDir,'textbook/models/gpt-3/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const name of ['gpt3-demonstrations-frame-171.png','10-gpt3-fine-tuning.gif',
      '07-gpt3-processing-transformer-blocks.gif','tokens-through-stack-frame-385.png',
      'context-window-frame-229.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['Повторное использование демонстраций','10 000 входных','2080 новых позиций',
      '72 пары','57','Перевод: направление и параллельные примеры','SacreBLEU',
      'Преобразования слов и новое значение','0.44%','Генерация новостей и оценка человеком',
      '49–54%','48–57%','фильтр оставил 2944 из 5153','На них accuracy составила 86.0%',
      'Остальные 2209 примеров с пересечениями имеют accuracy 86.9%'])
      expect(text).toContain(fact);
    expect(html).toContain('href="/sources/papers/gpt-3-0/"');
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/gpt-2\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/t5\/"/);
  });
  it('renders corrected GPT-3 paper-note comparisons without false historical priorities',async()=>{
    const html=await readFile(join(distDir,'sources/papers/gpt-3-0/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'').replace(/\s+/g,' ');
    for(const fact of ['31 автор','28 мая 2020','22 июля 2020','разница 5.7 пункта',
      '10.2 процентного пункта','2944 чистых примера с accuracy 86.0%',
      '2209 пересекающихся с accuracy 86.9%','82.8%','звёздочкой'])
      expect(text).toContain(fact);
    expect(html).toContain('href="/textbook/models/gpt-3/"');
    expect(text).not.toContain('45 авторов');
    expect(text).not.toContain('впервые задачи задаются');
    expect(html).not.toContain('katex-error');
  });
  it('renders GPT-2 cache contracts, the six originals and complete model examples',async()=>{
    const html=await readFile(join(distDir,'textbook/models/gpt-2/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'');
    for(const name of ['gpt2-sizes-hyperparameters-3.png','gpt2-input-embedding-positional-encoding-3.png',
      'gpt2-transformer-block-vectors-2.png','gpt2-output-projection-2.png',
      'gpt2-autoregression-2.gif','gpt2-self-attention-qkv-3-2.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const fact of ['class TinyGPT2','семь позиций вместо четырёх','индекс 7 вместо 4',
      'Последний выбранный ID уже входит','18','874','368','124 439 808','two-shot'])
      expect(text).toContain(fact);
    expect(html).toContain('href="/licenses/openai-gpt2-Modified-MIT.txt"');
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/gpt-1\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/gpt-3\/"/);
  });
  it('renders corrected GPT-2 paper-note protocols and released model sizes',async()=>{
    const html=await readFile(join(distDir,'sources/papers/gpt-2-0/index.html'),'utf8');
    for(const fact of ['124M','355M','774M','52.66%','93.30%','18.34','42.16',
      '29.34/8.27/26.58','демонстрации в контексте','93.45','93.30'])
      expect(html).toContain(fact);
    expect(html).toContain('href="/textbook/models/gpt-2/"');
    expect(html).not.toContain('ROUGE-L ~15');
    expect(html).not.toContain('katex-error');
  });
  it('renders GPT-1 gradients, the full candidate example and the original figure',async()=>{
    const html=await readFile(join(distDir,'textbook/models/gpt-1/index.html'),'utf8');
    const text=html.replace(/<[^>]*>/g,'');
    expect(html).toMatch(/<img[^>]+src="[^"]*\/gpt1-figure1\.png"/);
    for(const fact of ['head.grad','scores.retain_grad','token_loss.retain_grad',
      '0.020833','0.718351','2.085067','Неверный кандидат тоже участвует в LM-обучении',
      'Скрытое состояние заново вычисляется'])
      expect(text).toContain(fact);
    expect(html).toContain('href="/licenses/openai-gpt1-MIT.txt"');
    expect(html.match(/<pre/g)).toHaveLength(5);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/bert-roberta-deberta\/"/);
    expect(html).toMatch(/<strong>Далее:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/gpt-2\/"/);
  });
  it('renders the corrected GPT-1 paper note and its link to the full chapter',async()=>{
    const html=await readFile(join(distDir,'sources/papers/gpt-1-0/index.html'),'utf8');
    expect(html).toContain('href="/textbook/models/gpt-1/"');
    for(const fact of ['не было изобретено','85.8%','59.9','75.0','69.1',
      '56.0 против 61.7','82.3 F1 против 86.0','few-shot с демонстрациями'])
      expect(html).toContain(fact);
    expect(html).not.toContain('katex-error');
  });
  it('renders BERT training, both source excerpts and nine original figures',async()=>{
    const html=await readFile(join(distDir,'textbook/models/bert-roberta-deberta/index.html'),'utf8');
    for(const name of ['bert-base-bert-large-encoders.png','bert-figure1.png','bert-figure2.png',
      'BERT-language-modeling-masked-lm.png','bert-next-sentence-prediction.png',
      'bert-figure4.png','bert-feature-extraction-contextualized-embeddings.png',
      'spanbert-boundary-objective.png','spanbert-span-lengths.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.','\\.')+'"'));
    for(const id of ['архитектура-и-формы-тензоров','mlm-как-получить-двунаправленный-обучающий-сигнал',
      'вторая-цель-bert-отношения-между-сегментами','данные-и-рецепт-bert',
      'один-encoder--разные-способы-чтения-выхода','как-читать-результаты-bert',
      'roberta-когда-новый-результат-даёт-не-новая-архитектура',
      'deberta-разделить-содержание-и-относительное-положение',
      'ограничения-и-наследие','источники-и-дальнейшее-чтение'])
      expect(html).toContain('id="'+id+'"');
    for(const fact of ['TinyBertPretraining','emd_context_layer','disentangled',
      'mlm_input_grad','0.118682','first read','second read','2.390310','89.875',
      '3.674235','Microsoft Corporation','microsoft-deberta-MIT.txt'])
      expect(html).toContain(fact);
    expect(html.match(/<pre/g)).toHaveLength(11);
    expect(html).not.toContain('katex-error');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/transformer\/encoder-decoder-patterns\/"/);
    expect(html).toMatch(/<strong>Дальше:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/gpt-1\/"/);
  });
  it('renders the same grouped-masking and SOP additions in both BERT locales', async()=>{
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir,prefix+'textbook/models/bert-roberta-deberta/index.html'),'utf8');
      for (const anchor of ['whole-word-masking','spanbert-boundary-objective',
        'spanbert-span-sampling','sentence-order-prediction'])
        expect(html).toContain('id="'+anchor+'"');
      for (const fact of ['choose_word_targets','direct_sbo_gradient','same_offset_logits',
        'clamped_geometric_mean','assemble_sop','next_sentence_labels',
        'google-bert-eedf5716-Apache-2.0.txt','google-albert-b772393d-Apache-2.0.txt'])
        expect(html).toContain(fact);
      expect(html.match(/<pre/g)).toHaveLength(11);
      expect(html).not.toContain('katex-error');
    }
  });
  it('renders corrected RoBERTa and DeBERTa paper notes with links to the full chapter', async()=>{
    const deberta=await readFile(join(distDir,'sources/papers/deberta/index.html'),'utf8');
    const roberta=await readFile(join(distDir,'sources/papers/roberta/index.html'),'utf8');
    for(const html of [deberta,roberta])
      expect(html).toContain('href="/textbook/models/bert-roberta-deberta/"');
    for(const fact of ['81.3/78.0','91.8/85.8','1.3 пункта','DeBERTa-MT','89.875'])
      expect(deberta).toContain(fact);
    for(const fact of ['78.3/78.7','84.3/84.0','92.5/92.9','начальному опыту на 100K шагов'])
      expect(roberta).toContain(fact);
  });
  it('renders six architecture originals, all four examples and 27 previous anchors', async () => {
    const html = await readFile(join(distDir,'textbook/transformer/encoder-decoder-patterns/index.html'),'utf8');
    for (const name of ['t5-figure1.png','t5-figure2.png','t5-figure3.png',
      't5-figure4.png','bert-figure4.png','BERT-language-modeling-masked-lm.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.', '\\.')+'"'));
    for (const fact of ['1.832581','0.880797','11.125','12.25','corrupt_spans',
      'full after append','new causal query','cross_attention_cache',
      'decoder_ids','two_layers','CC BY 4.0','CC BY-NC-SA 4.0'])
      expect(html).toContain(fact);
    for (const id of ["один-пример-на-всю-главу","главный-переключатель-матрица-видимости","encoder-only-построить-контекстное-представление","как-bert-учится-без-готовой-разметки","что-выходит-из-encoder","почему-encoder-неудобен-для-обычной-генерации","decoder-only-предсказать-следующий-токен","почему-из-next-token-prediction-получается-general-purpose-модель","минимальный-causal-attention-в-коде","encoderdecoder-прочитать-один-текст-написать-другой","один-шаг-перевода","teacher-forcing","t5-все-задачи-как-text-to-text","почему-слово-decoder-сбивает-с-толку","три-маски-можно-объединить-в-одной-модели","что-выбрать-на-практике","сравнение-без-рекламных-упрощений","ошибки-которые-стоит-научиться-замечать","практика-без-большого-gpu","1-нарисовать-маски","2-сравнить-representations","3-прочитать-настоящий-код","контроль-понимания","литература-и-дополнительные-материалы","основные-papers","лучшие-объяснения-и-код","связанные-заметки-базы"]) expect(html).toContain('id="'+id+'"');
    expect(html.match(/<pre/g)).toHaveLength(8);
    expect(html).not.toContain('katex-error');
  });
  it('routes the architecture introduction back to Transformer and forward to BERT', async () => {
    const html = await readFile(join(distDir,'textbook/transformer/encoder-decoder-patterns/index.html'),'utf8');
    expect(html).toMatch(/<strong>Назад:<\/strong>[\s\S]{0,300}href="\/textbook\/transformer\/full-transformer\/"/);
    expect(html).toMatch(/<strong>Дальше:<\/strong>[\s\S]{0,300}href="\/textbook\/models\/bert-roberta-deberta\/"/);
  });
  it('renders both complete Transformer models, seven originals and preserved headings', async () => {
    const html = await readFile(join(distDir,'textbook/transformer/full-transformer/index.html'),'utf8');
    for (const name of ['Transformer_encoder.png','The_transformer_encoder_decoder_stack.png',
      'transformer_resideual_layer_norm_2.png','transformer_resideual_layer_norm_3.png',
      'transformer_decoder_output_softmax.png','transformer_decoding_2.gif',
      'gpt-2-transformer-xl-bert-3.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.', '\\.')+'"'));
    for (const fact of ['TinyTranslation','TinyCausalLM','EncoderBlock','DecoderBlock',
      '0.349012','0.549012','0.365148','SublayerConnection','CC BY-NC-SA 4.0',
      '9.786662','12.110340','9.352585','8.497538','7.482068',
      'per-target loss','shared PAD row gradient norm','A Transformer block contains two'])
      expect(html).toContain(fact);
    for (const id of ['1-главная-декомпозиция','2-остаточный-поток','3-feed-forward-network',
      '4-информация-о-позиции','5-оригинальный-encoder-block','6-оригинальный-decoder-block',
      '7-обучение-и-генерация','8-три-архитектурные-ветви','encoder-only-bert',
      'decoder-only-gpt','modern-decoder-only-llama','9-архитектурный-diff',
      '10-архитектурное-решение-и-эмпирический-рецепт--не-одно-и-то-же',
      '11-почему-transformer-вытеснил-рекуррентную-основу',
      '12-минимальный-современный-causal-block','13-как-читать-реальный-код',
      '14-проверка-форм-тензоров','краткие-итоги','источники-и-интерактивы',
      'первичные','код-и-объяснения','architecture-lineage','architecture-evidence-boundary'])
      expect(html).toContain('id="'+id+'"');
    expect(html).not.toContain('katex-error');
  });
  it('renders complete positional mechanisms, six source figures and stable section anchors', async () => {
    const html = await readFile(join(distDir,'textbook/transformer/positional-information/index.html'),'utf8');
    for (const name of ['transformer_positional_encoding_vectors.png',
      'd2l-sinusoidal-curves.svg','d2l-positional-heatmap.svg',
      'shaw-relative-figure1.png','roformer-figure1.png','alibi-original.jpeg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.', '\\.')+'"'));
    for (const fact of ['0.804430','18.517688','0.576117','32.344406','0.960170',
      't5_bucket','apply_rope','alibi_slopes','reset-position error',
      '3.061223','1.580556','incremental RoPE outputs','double-rotated cache output',
      'CC BY-NC-SA 4.0','CC BY-SA 4.0','CC BY-NC-ND 4.0'])
      expect(html).toContain(fact);
    for (const id of ['отправная-точка-где-position-входит-в-attention',
      'абсолютные-позиции-адрес-каждой-строки','обучаемая-таблица','синусоидальное-кодирование',
      'относительные-представления-shaw-расстояние-становится-частью-связи',
      't5-relative-position-bias-один-скаляр-на-bucket-и-голову',
      'rope-позиция-как-поворот-query-и-key','маленький-численный-пример-rope',
      'alibi-линейный-штраф-за-дальность','один-пример-шесть-разных-вмешательств',
      'сравнение-методов','экстраполяция-что-именно-следует-проверять',
      'практические-ошибки-которые-не-видны-по-shapes','краткие-итоги',
      'источники-и-дальнейшее-чтение','rope-relative-position-derivation',
      'rope-и-kv-cache-позиция-сохраняется-вместе-с-ключом'])
      expect(html).toContain('id="'+id+'"');
    expect(html).not.toContain('katex-error');
  });
  it('renders device-placement semantics beside the original CUDA stream figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-systems/gpu-cuda-memory-hierarchy/index.html'), 'utf8');
    expect(html).toContain('id="torch-device-placement"');
    for (const text of ['source_cpu', 'on_device', 'explicit_cpu', 'same_cpu',
      'independent_cpu', 'CUDA transfer executed', 'CPU-ветка',
      'nvidia-same-stream.png', 'nvidia-two-streams.png']) expect(html).toContain(text);
    expect(html).toContain('https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.to.html');
    expect(html).toContain('datetime="2026-10-29T00:00:00.000Z"');
    expect(html).not.toContain('class="katex-error');
  });

  it('renders masking/head examples, source figures and existing chapter anchors', async () => {
    const html = await readFile(join(distDir,'textbook/transformer/masking-heads-shapes/index.html'),'utf8');
    for (const name of ['causal-mask-detailed.png', 'qkv-for-heads.png',
      'transformer_attention_heads_qkv.png','transformer_attention_heads_z.png',
      'transformer_attention_heads_weight_matrix_o.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.', '\\.')+'"'));
    for (const fact of ['12.689414','11.546979','0.909969','32','-6',
      'absolute-position mask', 'upper-left causal mask','two-query absolute mask',
      'two-query outputs','block_unmasked','causal_lower_right','masked_attention',
      'MultiHeadSelfAttention','loss.backward','CC BY 4.0','CC BY-NC-SA 4.0'])
      expect(html).toContain(fact);
    for (const namedAxisExample of ['named_scores', 'norm_sq',
      'scores of first query', 'squared row norms',
      'https://einops.rocks/api/rearrange/', 'https://einops.rocks/api/einsum/',
      'https://einops.rocks/api/reduce/']) expect(html).toContain(namedAxisExample);
    for (const id of ['маска-меняет-множество-доступных-ключей',
      'причинная-маска-не-подсматривать-ответ','padding-mask-не-читать-заполнители',
      'зачем-одной-позиции-несколько-голов','split-и-merge-одна-операция-две-записи',
      'полная-трассировка-форм','broadcasting-масок','ошибки-которые-выглядят-правдоподобно',
      'минимальная-проверка-реализации','источники-и-визуальные-продолжения',
      'named-attention-shapes','einops-attention-shapes','shape-evidence'])
      expect(html).toContain('id="'+id+'"');
    expect(html).not.toContain('katex-error');
  });
  it('renders projected self-attention, complete examples and four original figures', async () => {
    const html = await readFile(join(distDir,'textbook/transformer/self-attention/index.html'),'utf8');
    for (const name of ['transformer_self_attention_vectors.png',
      'self-attention-matrix-calculation-2.png', 'self-attention-output.png',
      'cnn-rnn-self-attention.svg'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/'+name.replaceAll('.', '\\.')+'"'));
    for (const fact of ['2.046556','1.838140','0.196612','1.758725',
      '2.093112','0.436612','0.718517','0.058699',
      'one SGD step', 'torch.testing', 'CC BY-NC-SA 4.0', 'CC BY-SA 4.0'])
      expect(html).toContain(fact);
    for (const id of ['зачем-слову-смотреть-на-другие-слова',
      '1-самый-простой-канал-среднее-прошлого',
      '2-сделаем-веса-зависимыми-от-содержимого','3-почему-нужен-отдельный-value',
      'всё-вычисление-на-двух-рисунках','4-численный-пример-который-можно-проверить-вручную',
      'шаг-a-dot-products','шаг-b-scaling','шаг-c-softmax','шаг-d-weighted-sum-values',
      '5-зачем-делить-на-dksqrtd_kdk','6-минимальная-реализация-одной-головы',
      'шаг-a-скалярные-произведения','шаг-b-масштабирование',
      'шаг-d-взвешенная-сумма-значений',
      'обратный-проход-на-том-же-примере','код-прямого-и-обратного-прохода',
      'что-легко-перепутать','different-weights-the-same-output',
      'лаборатория-по-карпати','источники'])
      expect(html).toContain('id="'+id+'"');
    expect(html).toContain('different weights:');
    expect(html).toContain('same output:');
    expect(html).toContain('https://arxiv.org/html/1908.04211v3#S3');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(4);
    expect(html).not.toContain('katex-error');
  });

  it('renders recurrent-attention mechanisms, full examples and all eight source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/transformer/from-seq2seq-to-transformer/index.html'), 'utf8');
    for (const name of ['bottleneck-min.png', 'bahdanau_examples-min.png',
      'd2l-attention-output.svg', 'distill-rnn_read.svg', 'seq2seq-attention.svg',
      'luong-global.png', 'luong-local.png', 'luong-input-feeding.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replaceAll('.', '\\.') + '"'));
    for (const fact of ['1.955285', '0.000560', '0.969', 'AdditiveAttention',
      '192', '768', 'detach()', 'EMNLP 2015',
      'AttentiveSeq2Seq', 'context_normalized', 'input feeding', 'CC BY-NC-SA 3.0'])
      expect(html).toContain(fact);
    for (const id of ['перевод-как-условная-языковая-модель',
      '1-до-attention-один-вектор-на-всё-предложение', '2-оставим-не-резюме-а-память',
      '2-позиционные-представления-и-чтение-памяти',
      '2-представления-исходных-позиций-и-чтение-памяти',
      '3-проследим-один-перевод', '4-bahdanau-и-luong--похожий-принцип-разные-сборки',
      '5-что-именно-улучшил-attention', '6-почему-этого-всё-ещё-недостаточно',
      '7-три-термина-которые-нельзя-смешивать', 'мини-реализация-recurrent-attention',
      'краткие-итоги', 'источники-и-хорошие-продолжения', 'как-получить-веса',
      'динамический-доступ', 'короткий-путь-для-информации-и-градиента',
      'переменная-длина-памяти', 'наблюдаемый-alignment',
      'два-объяснения-которые-стоит-прочитать-в-оригинале', 'первичные', 'объяснения'])
      expect(html).toContain('id="' + id + '"');
    expect(html).not.toContain('katex-error');
  });

  it('renders the complete seq2seq examples, seven source SVGs and preserved sections', async () => {
    const html = await readFile(join(distDir, 'textbook/recurrent-networks/seq2seq-bottleneck/index.html'), 'utf8');
    expect(html).toContain('0.000583');
    expect(html).toContain('pack_padded_sequence');
    expect(html).toContain('enumerate_finished');
    expect(html).toContain('exposure bias');
    expect(html).not.toContain('katex-error');
    for (const id of [
      'условная-языковая-модель', 'формы-на-одном-примере',
      'почему-фиксированный-контекст-становится-узким-местом',
      'декодирование--не-обучение', 'мост-к-attention',
      'ограничения-и-частые-смешения', 'источники-и-дальнейшее-чтение'
    ]) expect(html).toContain('id="' + id + '"');
    for (const name of ['seq2seq', 'seq2seq-details', 'seq2seq-predict',
      's2s-prob1', 's2s-prob2', 'beam-search', 'seq2seq-attention']) {
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name + '\\.svg"'));
    }
  });

  it('renders activation mechanisms, faithful source figures and the output-loss exception', async () => {
    const html = await readFile(join(distDir, 'textbook/neural-networks/activation-functions/index.html'), 'utf8');
    for (const figure of ['d2l-sigmoid.svg', 'd2l-sigmoid-gradient.svg', 'd2l-tanh.svg',
      'd2l-tanh-gradient.svg', 'd2l-relu.svg', 'd2l-relu-gradient.svg',
      'gelu-figure-1.png', 'cs336-silu-vs-relu.png'])
      expect(html.includes('/' + figure), 'missing original activation figure: ' + figure).toBe(true);
    for (const anchor of ['sigmoid', 'tanh', 'relu', 'elu', 'gelu', 'silu--swish',
      'сравнение-по-поведению', 'активация-и-инициализация-связаны',
      'где-активация-не-нужна', 'краткие-итоги', 'источники'])
      expect(html.includes('id="' + anchor + '"'), 'missing stable activation anchor: ' + anchor).toBe(true);
    for (const fact of ['binary_cross_entropy_with_logits', '-0.993307', '0{,}006648',
      'Leaky ReLU', '0{,}6875', 'SwiGLU', '2726505c4767eed99dfd89ce0f836fbca44d6a6d',
      'Левый и правый наклоны', 'В минимуме', 'производная по', '0.655293'])
      expect(html.includes(fact), 'missing activation explanation: ' + fact).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('<p>![')).toBe(false);
    expect((html.match(/class="katex-display"/g) ?? []).length).toBeGreaterThanOrEqual(20);
    const license = await readFile(join(distDir, 'licenses/gelus-MIT.txt'), 'utf8');
    expect(license.includes('Copyright (c) 2016 hendrycks')).toBe(true);
  });
  it('renders the reviewed MLP calculation, original figures and stable anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/neural-networks/mlp/index.html'), 'utf8');
    for (const figure of ['cs231n-mlp-layered.jpeg', 'cs231n-depth-width.jpeg',
      'cs231n-decision-regions.jpeg', 'moon-mlp.png'])
      expect(html.includes('/' + figure), 'missing MLP source figure: ' + figure).toBe(true);
    for (const anchor of ['один-нейрон', 'слой-считает-много-нейронов-одновременно',
      'зачем-нужна-нелинейность', 'xor-как-минимальный-контрпример-линейной-модели',
      'что-именно-обучается', 'ширина-глубина-и-число-параметров',
      'mlp-внутри-transformer', 'краткие-итоги', 'источники'])
      expect(html.includes('id="' + anchor + '"'), 'missing stable MLP anchor: ' + anchor).toBe(true);
    for (const fact of ['0{,}126928', '0{,}098018', '-0{,}298007', 'optimizer.step',
      'weight.copy_', 'Глубина одинакова', 'Positionwise Feed-Forward Networks',
      'grad', 'hinge loss', 'MIT License'])
      expect(html.includes(fact), 'missing worked MLP explanation: ' + fact).toBe(true);
    for (const mechanism of ['model-state-parameters-buffers-and-submodules',
      'registration-does-not-depend-on-a-gradient', 'a-list-of-layers-must-register-its-contents',
      'saving-values-and-rebuilding-the-computation', 'evaluation-mode-and-gradient-recording',
      'CustomLayer', 'ModuleList', 'probe_optimizer', 'checkpoint_bytes', 'weights_only',
      'does not encode the ReLU', 'does not disable gradient recording',
      '/licenses/pytorch-v2.14.1-LICENSE.txt'])
      expect(html.replace(/\s+/g, ' ').includes(mechanism), 'missing module-state explanation: ' + mechanism).toBe(true);
    expect(html.includes('Слева показан нейрон как вычислительный узел')).toBe(false);
    expect(html.includes('Сети различной глубины и ширины')).toBe(false);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('<p>![')).toBe(false);
    expect((html.match(/class="katex-display"/g) ?? []).length).toBeGreaterThanOrEqual(15);
  });
  it('renders complete module hooks and distinct input/output/weight derivatives in both locales',async()=>{
    for(const prefix of ['', 'en/']) {
      const html=await readFile(join(distDir,prefix+'textbook/neural-networks/mlp/index.html'),'utf8');
      for(const fact of ['observing-a-layer-forward-and-backward-hooks','one-layer-three-different-derivatives',
        'when-the-backward-hook-fires','inspect_linear','capture_backward','input_gradient',
        'output_gradient','no_grad_records','bypass_records','13.5','CC BY-NC-SA 4.0'])
        expect(html).toContain(fact);
      expect(html).toContain('frameworks.qmd#L3871');
      expect(html).not.toContain('class="katex-error');
      if(prefix) expectCanonicalEditionNotice(html);
    }
  });
  it('renders formal mathematics mechanisms, source figures and checked proof examples', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/formal-math/index.html'), 'utf8');
    for (const figure of ['leanstar-pipeline.png', 'leanstar-search-sampling.png',
      'm09-p97.png', 'm10-p78.png', 'm11-p37.png'])
      expect(html.includes('/' + figure), 'missing original formal-math figure: ' + figure).toBe(true);
    for (const mechanism of ['swap_and', 'larger_exists', 'no_largest',
      'SMT_VERIF', 'associated_of_dvd_dvd', 'Chain-of-States',
      'compile_one_instr', 'sample_with_continuation', 'insufficient_stack',
      'Coq.Lists.List', 'app_nil_r', '46,3%', '31,65%', 'CC BY 4.0'])
      expect(html.includes(mechanism), 'missing formal-math explanation: ' + mechanism).toBe(true);
    for (const anchor of ['один-шаг-lean-proof-state-tactic-kernel-check',
      'три-разрыва-theorem-statement-proof-steps-and-semantic-equivalence',
      'lean-star-thought-before-tactic-and-expert-iteration',
      'copra-stateful-backtracking-with-prover-feedback', 'воспроизводимый-research-workflow'])
      expect(html.includes('id="' + anchor + '"'), 'missing preserved anchor: ' + anchor).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('<p>![')).toBe(false);
  });
  it('renders coding, web and computer-use mechanisms with original figures and examples', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/coding-web-computer-use/index.html'), 'utf8');
    for (const figure of ['taco-cota-examples.png', 'm05-p17-react-aci.png',
      'm06-p18-visualwebarena-shopping.png', 'm06-p60-search-backtracking.png',
      'm07-p11-osworld-contract.png', 'm07-p72-aguvis-two-stage-training.png'])
      expect(html.includes('/' + figure), 'missing original figure: ' + figure).toBe(true);
    for (const original of [
      'https://arxiv.org/html/2403.17134v2/states_diagram2.drawio.svg',
      'https://arxiv.org/html/2409.16165v1/EnIGMA_Figure1.png',
      'https://arxiv.org/html/2412.09605v2/images/part2_v6.png']) {
      expect(html.includes('src="' + original + '"')).toBe(true);
      expect(html.includes('href="' + original + '"')).toBe(true);
    }
    for (const name of ['best_first_fixture', 'parse_assignment', 'express_hypothesis',
      'collect_more_information', 'CC BY 4.0', '14,6%', '16,7%', '6 963',
      'heap-buffer-overflow', 'Step Accuracy', 'None'])
      expect(html.includes(name), 'missing mechanism: ' + name).toBe(true);
    for (const anchor of ['свободный-loop-aci-и-процедурный-workflow',
      'html-accessibility-tree-screenshot-и-set-of-marks',
      'почему-реактивный-следующий-клик-ломается-на-длинном-горизонте',
      'osworld-resettable-vm-executable-action-and-final-state-grader',
      'taco-совместно-обучать-reasoning-и-action-tokens',
      'минимальный-parser-repair-с-наблюдаемым-результатом'])
      expect(html.includes('id="' + anchor + '"'), 'missing preserved anchor: ' + anchor).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('<p>![')).toBe(false);
  });
  it('renders agent evaluation mechanisms, original figures and worked examples', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/evaluation/index.html'), 'utf8');
    for (const figure of ['swe-bench-construction.png', 'swe-bench-evaluation.png',
      'tau-bench-task-components.png', 'paperbench-rubric-tree.png', 'agentbench-environments-results.png'])
      expect(html.includes('/' + figure), 'missing evaluation figure: ' + figure).toBe(true);
    for (const original of [
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/bd42e7b2f3e9bb5218142796d3ede4816588dec0-4584x2834.png',
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/0205b36f9639fc27f2f6566f73cb56b06f59d555-4584x2580.png',
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/3ddac5be07a0773922ec9df06afec55922f8194a-4584x2580.png']) {
      expect(html.includes('src="' + original + '"')).toBe(true);
      expect(html.includes('href="' + original + '"')).toBe(true);
    }
    for (const name of ['grade_booking', 'estimate_pass_at_k', 'resampled_means', 'weighted_score',
      'FAIL_TO_PASS', 'PASS_TO_PASS', 'reproduce.sh', 'CC BY 4.0', '/licenses/human-eval-b9453e8f-MIT.txt'])
      expect(html.includes(name), 'missing evaluation mechanism: ' + name).toBe(true);
    for (const anchor of ['срок-жизни-benchmark-и-несмещённый-passk', '9-модель-судья',
      '10-безопасность-оценивается-как-инвариант', '12-сопоставимость-тестовой-и-рабочей-среды'])
      expect(html.includes('id="' + anchor + '"'), 'missing preserved anchor: ' + anchor).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('<p>![')).toBe(false);
  });
  it('renders memory retrieval, planning and orchestration with original figures', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/memory-planning-orchestration/index.html'), 'utf8');
    for (const figure of ['tot-game24-search.png', 'reflexion-loop.png',
      'memgpt-figure3-memory-hierarchy.png', 'webdreamer-planning-paradigms.png',
      'm03-p31-hipporag-index-and-retrieval.png', 'm03-p70-webdreamer-worked-example.png',
      'm01-p64-self-debugging-feedback.png', 'm01-p67-self-correction-no-oracle.png',
      'm01-p69-debate-vs-self-consistency.png'])
      expect(html.includes('/' + figure), 'missing original figure: ' + figure).toBe(true);
    for (const original of [
      'https://arxiv.org/html/2304.03442v2/figures/figure_reflection6.png',
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png',
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png']) {
      expect(html.includes('src="' + original + '"')).toBe(true);
      expect(html.includes('href="' + original + '"')).toBe(true);
    }
    for (const name of ['minmax', 'ppr_step', 'try_candidates', 'accept_result',
      'passed_given_cases', 'accepted_for_review', 'request_heartbeat', 'CC BY 4.0'])
      expect(html.includes(name), 'missing mechanism: ' + name).toBe(true);
    for (const name of ['method.png', 'sequencing.png', 'nut-assembly.png'])
      expect(html.includes('/planseqlearn-2026-10-26/' + name)).toBe(true);
    for (const original of ['Planning Module: Zero-Shot High-level Planning',
      'Sequencing Module: Vision-based Plan Tracking', 'Efficiently Learning Local Control',
      'Stage Termination Details', '0{,}6561']) expect(html.includes(original)).toBe(true);
    expect(html.includes('id="plan-seq-learn"')).toBe(true);
    expect(html.includes('alt="PSL: языковой план, движение к объекту и локальное управление"')).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('<p>![')).toBe(false);
    for (const anchor of ['1-память-находится-вне-весов', '5-извлечение-не-только-nearest-neighbors',
      'реактивный-агент-поиск-в-реальной-среде-и-model-based-planning',
      'когда-более-сильной-модели-нужен-менее-жёсткий-scaffold',
      '7-рефлексия-полезна-только-через-наблюдаемую-обратную-связь', '10-контракт-между-агентами'])
      expect(html.includes('id="' + anchor + '"'), 'missing preserved anchor: ' + anchor).toBe(true);
  });
  it('renders harness state, context budgets, original figures and preserved deep links', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/harness-context-engineering/index.html'), 'utf8');
    for (const name of ['swe-agent-prompt-workflow.png', 'lost-middle-task-controls.png', 'lost-middle-position-effect.png'])
      expect(html.includes('/' + name), 'missing harness figure: ' + name).toBe(true);
    const unescaped = html.replace(/&(?:amp|#x26|#38);/gi, '&');
    for (const original of [
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/faa261102e46c7f090a2402a49000ffae18c5dd6-2292x1290.png',
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/0442fe138158e84ffce92bed1624dd09f37ac46f-2292x1288.png',
      'https://mintcdn.com/allhandsai/hwL6uO0ZqM-lmlYG/sdk/guides/assets/condenser_overview_light_mode.png?fit=max&auto=format&n=hwL6uO0ZqM-lmlYG&q=85&s=dc85afa9f2821b1f7a305ad1406ab3d9']) {
      expect(unescaped.includes('src="' + original + '"')).toBe(true);
      expect(unescaped.includes('href="' + original + '"')).toBe(true);
    }
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('<p>![')).toBe(false);
    for (const value of ['history_capacity', 'common_prefix_length', 'valid_checkpoint',
      'update_reminders', 'run_episode', 'tested_revision', 'Appendix C', 'CC BY 4.0'])
      expect(html.includes(value), 'missing harness explanation: ' + value).toBe(true);
    for (const anchor of ['5-контекст-со-временем-портится',
      '6-compaction-как-преобразование-состояния', '7-reminder-tokens--не-особый-вид-токена',
      '1-из-чего-состоит-один-шаг', '10-наблюдаемость-и-доказательство-завершения'])
      expect(html.includes('id="' + anchor + '"'), 'missing stable harness anchor: ' + anchor).toBe(true);
  });
  it('renders the tool-use mechanisms, original ReAct panels and executable fixtures', async () => {
    for (const locale of ['', 'en/']) {
    const html = await readFile(join(distDir, locale, 'textbook/agents/tool-use/index.html'), 'utf8');
    for (const name of ['react-hotpotqa-primary.png', 'react-alfworld-primary.png'])
      expect(html.includes('/' + name), 'missing ReAct panel: ' + name).toBe(true);
    for (const original of ['https://react-lm.github.io/files/diagram.png',
      'https://www-cdn.anthropic.com/images/4zrzovbb/website/58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png']) {
      expect(html.includes('src="' + original + '"')).toBe(true);
      expect(html.includes('href="' + original + '"')).toBe(true);
    }
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('<p>![')).toBe(false);
    for (const value of ['continuation_loss', '1.3025', 'idempotency_parameter_mismatch',
      'run_episode', 'readOnlyHint', 'CC BY 4.0', '2025-11-25'])
      expect(html.includes(value), 'missing tool-use explanation: ' + value).toBe(true);
    for (const anchor of ['где-начинается-агент-модель-память-действие-и-среда',
      'два-способа-проектировать-агента-llm-first-и-agent-first',
      'как-toolformer-получает-обучающие-примеры', '4-наблюдение-должно-отвечать-на-следующий-вопрос',
      '5-ошибка--часть-протокола', '8-как-проверять-tool-use'])
      expect(html.includes('id="' + anchor + '"'), 'missing stable tool-use anchor: ' + anchor).toBe(true);
    for (const anchor of ['react-prompting-learning','toolformer-curation-evaluation'])
      expect(html).toContain('id="'+anchor+'"');
    for (const file of ['react-cot-sampling.png','react-prompt-vs-finetune.png'])
      expect(html).toContain('agent-reconciliation-2026-10-30/'+file);
    for (const example of ['insertion_sites','fallback_to_react','calls_already_made'])
      expect(html).toContain(example);
    expect(html.match(/<pre\b/g)).toHaveLength(10);
    }
  });
  it('renders Chameleon mechanisms, five licensed source crops and the original external VQ figure', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/unified-sequence-chameleon/index.html'), 'utf8');
    for (const name of ['chameleon-architecture.png', 'chameleon-mixed-document.png',
      'chameleon-norm-qk-dropout.png', 'chameleon-training-ablations.png', 'chameleon-human-evaluation.png'])
      expect(html.includes('/' + name), 'missing Chameleon figure: ' + name).toBe(true);
    const original = 'https://arxiv.org/html/1711.00937v2/figures/Figure1_9.png';
    expect(html.includes('src="' + original + '"')).toBe(true);
    expect(html.includes('href="' + original + '"')).toBe(true);
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('<p>![')).toBe(false);
    for (const value of ['encoder_grad', 'gemini_plus_score_from_figure', '58.75', '3552', 'CC BY 4.0'])
      expect(html.includes(value), 'missing Chameleon explanation: ' + value).toBe(true);
    for (const anchor of ['3-как-изображение-становится-дискретными-кодами',
      '4-одна-авторегрессионная-последовательность', '5-что-происходит-при-генерации',
      '6-почему-совместное-обучение-становится-нестабильным',
      '8-что-даёт-единый-словарь-а-чего-он-не-гарантирует'])
      expect(html.includes('id="' + anchor + '"'), 'missing stable Chameleon anchor: ' + anchor).toBe(true);
  });
  it('renders diffusion mechanisms, nine source figures and stable chapter anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/diffusion-flow-images/index.html'), 'utf8');
    for (const name of ['mit-cfg-mnist.png', 'mit-unet.png', 'dit-architecture.png',
      'mit-conditional-marginal.png', 'mit-conditional-marginal-trajectories.png',
      'mit-flow-training-result.png', 'sd3-mmdit.png', 'algorithms.png',
      'article-Figure3-1-1024x508.png'])
      expect(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"').test(html),
        'missing source figure: ' + name).toBe(true);
    expect(html.includes('class="katex')).toBe(true);
    expect(html.includes('class="katex-error')).toBe(false);
    expect(html.includes('<p>![')).toBe(false);
    for (const anchor of ['1-как-создаётся-учебный-пример',
      '2-почему-локальное-предсказание-позволяет-генерировать-целую-картинку',
      '3-latent-diffusion-denoising-не-в-rgb', '4-как-текст-влияет-на-пространственные-признаки',
      '5-u-net-dit-и-objective--разные-оси', '6-flow-matching-вместо-шума--скорость',
      '7-быстрая-проверка-понимания'])
      expect(html.includes('id="' + anchor + '"'), 'missing anchor: ' + anchor).toBe(true);
    for (const value of ['CC BY-NC-ND 4.0', 'mean_endpoint', 'importance_corrected', '154'])
      expect(html.includes(value), 'missing chapter content: ' + value).toBe(true);
    for (const url of ['https://hojonathanho.github.io/diffusion/assets/img/algorithms.png',
      'https://ommer-lab.com/wp-content/uploads/2022/08/article-Figure3-1-1024x508.png'])
      expect(html.includes('href="' + url + '"'), 'external figure must be enlargeable: ' + url).toBe(true);
  });
  it('includes one accessible source-figure viewer without replacing image-file links', async () => {
    const html = await readFile(join(distDir, 'textbook/retrieval/sparse-dense-hybrid/index.html'), 'utf8');
    expect(html.match(/<dialog\b[^>]*data-figure-viewer/g)).toHaveLength(1);
    expect(html).toContain('aria-labelledby="figure-viewer-title"');
    for (const control of ['data-viewer-minus', 'data-viewer-plus', 'data-viewer-fit', 'data-viewer-close'])
      expect(html).toContain(control);
    expect(html).toMatch(/<a\b[^>]*href="[^"]*\/ivf-build\.png"[^>]*>\s*<img/);
    expect(html).toContain('rel="prev"');
    expect(html).toContain('rel="next"');
  });
  it('links the 404 header to the configured publication root', async () => {
    const html = await readFile(join(distDir, '404.html'), 'utf8');
    const titleLink = html.match(/<a\b[^>]*class="[^"]*\bsite-title\b[^"]*"[^>]*>/)?.[0];
    const href = titleLink?.match(/\bhref="([^"]+)"/)?.[1];
    const base = (process.env.PUBLICATION_BASE_PATH ?? '').replace(/\/+$/, '');
    expect(href).toBe(`${base}/`);
    expect(html).not.toContain('class="sl-link-button');
  });

  // This walks the complete emitted book (currently 1,961 pages), not one page.
  // Disk-dependent wall time is not the acceptance criterion; every link is.
  it('has no broken emitted internal routes or fragments', async () => {
    await expect(findBrokenBuiltLinks(distDir)).resolves.toEqual([]);
  }, 30_000);
  it('does not publish local file edit links or workstation paths', async () => {
    const manifest = YAML.parse(await readFile(join(rootDir, 'publishing', 'navigation.yml'), 'utf8'));
    const routes: string[] = manifest.sections.flatMap(
      (section: { pages: Array<{ route: string }> }) => section.pages.map((page) => page.route)
    );
    for (const route of routes) {
      const outputRoute = route.endsWith('/index') ? route.slice(0, -'/index'.length) : route;
      const html = await readFile(join(distDir, outputRoute, 'index.html'), 'utf8');
      expect(html, route).not.toMatch(/file:\/\/|\/Users\//);
    }
  });

  it('renders formulas and source figures instead of exposing Markdown image syntax', async () => {
    const formula = await readFile(
      join(distDir, 'textbook', 'neural-networks', 'backpropagation', 'index.html'),
      'utf8'
    );
    const figure = await readFile(
      join(distDir, 'textbook', 'rag', 'full-pipeline', 'index.html'),
      'utf8'
    );

    expect(formula).toContain('class="katex');
    expect(figure).toContain('<img');
    for (const name of ['rag-sequence-token.png', 'hf-rag-evaluation-workflow.png',
      'm03-p31-hipporag-index-and-retrieval.png', 'lost-middle-position.png',
      'alce-citation-evaluation.png', 'openscholar-inference.png',
      'openscholar-training.png', 'hf-rag-settings-accuracy.png'])
      expect(figure).toContain(name);
    expect(figure).toContain('class="katex');
    expect(figure).not.toContain('class="katex-error');
    for (const anchor of [
      'почему-похожий-фрагмент-не-всегда-является-нужной-памятью',
      'hipporag-ассоциативное-извлечение-по-графу',
      'openscholar-литературный-поиск-как-итеративный-retrieval-and-synthesis-loop'
    ]) expect(figure).toContain('id="' + anchor + '"');
    expect(figure).not.toContain('<p>![');
  });

  it('renders AD modes, complete paper traces and both custom interfaces in root and EN fallback',async()=>{
    for(const prefix of ['', 'en/']) {
      const html=await readFile(join(distDir,prefix+'textbook/neural-networks/backpropagation/index.html'),'utf8');
      for(const filename of ['baydin-figure4-graph.png','baydin-table2-forward.png','baydin-table3-reverse.png',
        'cs231n-chain.svg','cs231n-gates.svg','cs231n-sigmoid.svg','computational-graph.svg'])
        expect(html).toContain(filename);
      for(const fact of ['MultiplyAddAD','setup_context','save_for_forward','curvature_loss',
        'expected_hvp','broadcast_inputs','check_forward_ad','gradgradcheck','CC BY 4.0'])
        expect(html).toContain(fact);
      expect(html).toContain('frameworks.qmd#L2324');
      expect(html).toContain('17-468.pdf#page=9');
      expect(html).not.toContain('class="katex-error');
      if(prefix) expectCanonicalEditionNotice(html);
    }
  });

  it('renders the complete JAX transformation example and its worked gradients in root and EN fallback',async()=>{
    for(const prefix of ['', 'en/']) {
      const html=await readFile(join(distDir,prefix+'textbook/neural-networks/backpropagation/index.html'),'utf8');
      for(const fact of ['Gradients as functions: grad, vmap and jit','fast_batched_grad',
        'compiled_per_example','mean_gradient','updated_params','column_targets',
        'wrong_gradient','torch_grad_fn','torch_per_example','1.1925','pytree'])
        expect(html).toContain(fact);
      expect(html).toContain('frameworks.qmd#L4003');
      expect(html).toContain('https://docs.jax.dev/en/latest/101/transformations.html');
      expect(html).toContain('not a speedup or a multi-GPU run');
      expect(html).not.toContain('class="katex-error');
      if(prefix) expectCanonicalEditionNotice(html);
    }
  });

  it('renders the JAX runtime examples and complete source diagram in root and explicit EN fallback',async()=>{
    for(const prefix of ['', 'en/']) {
      const html=await readFile(join(distDir,prefix+'textbook/ml-systems/gpu-kernels-triton/index.html'),'utf8');
      for(const fact of ['JAX: tracing, state and compiled control flow','fast_broken_count',
        'threaded_values','trace_signatures','hidden_gain','compiled_branch','sum_cumulative',
        'add_one_donated','single_mesh','psum.png','expected_columns',
        'four logical CPU devices','not four GPUs']) expect(html).toContain(fact);
      expect(html).toContain('/licenses/jax-dc4af684-Apache-2.0.txt');
      expect(html).toContain('jax.ref.new_ref.html');
      expect(html).not.toContain('class="katex-error');
      expect(html).not.toContain('jax.new_ref.html');
      if(prefix) expectCanonicalEditionNotice(html);
    }
    expect(await readFile(join(distDir,'licenses/jax-dc4af684-Apache-2.0.txt'),'utf8'))
      .toBe(await readFile(join(rootDir,'site/public/licenses/jax-dc4af684-Apache-2.0.txt'),'utf8'));
  });

  it('renders the micro-runtime source figure, complete fragments and lifetime example in both locales',async()=>{
    for(const prefix of ['', 'en/']) {
      const html=await readFile(join(distDir,prefix+'textbook/deployment/edge-federated/index.html'),'utf8');
      for(const fact of ['model-buffer-and-arena.png','microcontroller-inference-planning-a-tensor-arena',
        'lifetimes-determine-which-buffers-can-share-storage','check_arena_plan',
        'branch_offsets','phase_totals','RecordingMicroInterpreter','PrintAllocations',
        '9568','2048','not the RFC','No C++ build or board']) expect(html).toContain(fact);
      expect(html).toContain('/licenses/tflm-Apache-2.0.txt');
      expect(html).not.toContain('class="katex-error');
      if(prefix) expectCanonicalEditionNotice(html);
    }
    const license=await readFile(join(distDir,'licenses/tflm-Apache-2.0.txt'),'utf8');
    expect(license).toContain('Apache License');
  });

  it('renders the vision-encoder chapter with all six source figures and its stable sections', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/models/index.html'), 'utf8');
    for (const name of ['vit-figure1-patch-sequence.png', 'clip-figure1-training-and-zero-shot.png',
      'clip-objectives.png', 'clip-zero-shot-linear-probe.png', 'siglip-parallelism.png',
      'siglip-batch-size.png']) {
      expect(html).toContain(name);
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    }
    expect(html).toContain('class="katex');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    for (const anchor of ['1-изображение-как-тензор', '2-цена-более-высокого-разрешения',
      '4-как-из-clip-получается-zero-shot-классификатор',
      'clip-и-siglip-две-формы-контрастивного-обучения', '7-диагностические-проверки'])
      expect(html).toContain('id="' + anchor + '"');
  });

  it('renders the connector chapter with source figures, code and stable section anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/connectors-fusion/index.html'), 'utf8');
    for (const name of ['llava-figure1-projector.png', 'llava-onevision.png',
      'blip2-qformer.png', 'blip2-attention-masks.png', 'blip2-llm-connection.png',
      'flamingo-figure3-architecture.png', 'qwen3-vl.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    expect(html).toContain('class="katex');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    expect(html).toContain('predictor_positions');
    expect(html).toContain('requires_grad_');
    for (const anchor of ['3-blip-2-q-former-как-обучаемое-узкое-место',
      '4-flamingo-отдельная-визуальная-память', 'что-именно-вычисляет-resampler',
      '6-early-fusion-и-deep-fusion', '7-что-обучать-а-что-замораживать'])
      expect(html).toContain('id="' + anchor + '"');
  });

  it('renders resolution, packing, MRoPE examples and all six original figures', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/visual-tokenization-resolution/index.html'), 'utf8');
    for (const name of ['molmo-figure5-image-tokenization.png', 'llava-onevision-anyres.png',
      'internvl25-figure2-dynamic-tiles.png', 'qwen2-vl-architecture.png',
      'pixtral-vision-packing.png', 'qwen2vl-figure3-mrope.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    expect(html).toContain('class="katex');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    expect(html).toContain('rotate_three');
    expect(html).toContain('molmo_axis');
    expect(html).toContain('to_original');
    for (const anchor of ['1-бюджет-визуальных-токенов',
      'anyres-глобальный-кадр-и-локальные-плитки', '3-dynamic-high-resolution-в-internvl',
      'qwen2-vl-native-resolution-merger-и-mrope',
      '6-почему-одной-позиции-в-последовательности-недостаточно',
      '7-координаты-после-preprocessing', '8-практический-выбор'])
      expect(html).toContain('id="' + anchor + '"');
  });

  it('renders VLM training steps, loss masks, normalization and source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/training-data/index.html'), 'utf8');
    for (const name of ['internvl25-figure4-training-pipeline.png', 'molmo-figure1-data-and-capabilities.png',
      'llava-gen.png', 'llava-onevision-training.png', 'llava-onevision-transfer-s1.png',
      'llava-onevision-transfer-s2.png', 'qwen-vl-stages.png', 'qwen3-vl-pretraining.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    expect(html).toContain('class="katex');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    for (const name of ['active_rows', 'length_weighted_loss', 'frozen_head', '2661.12', 'SAPO'])
      expect(html).toContain(name);
    for (const anchor of ['1-что-происходит-на-последовательных-стадиях',
      '2-alignment-научить-llm-читать-визуальный-вход', '3-какие-параметры-размораживать',
      '4-как-устроен-loss-mask-в-диалоге', '5-какие-данные-создают-какие-способности',
      '6-как-смешивать-неодинаковые-наборы', '7-curriculum-по-разрешению-и-типам-задач',
      '8-post-training-должен-проверять-опору-на-изображение', '9-карточка-воспроизводимости'])
      expect(html).toContain('id="' + anchor + '"');
  });

  it('renders document mechanisms, source figures, metrics and preserved anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/documents-ocr-grounding/index.html'), 'utf8');
    for (const name of ['layoutlmv3-objectives.png', 'donut-overview.png',
      'pix2struct-screenshot-parsing.png', 'pix2struct-variable-resolution.png',
      'pix2struct-figure1-tasks.png', 'qwen2vl-figure11-dense-ocr.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    for (const text of ['anls_one', 'scale_crop', 'coordinate_space', 'CC BY-NC-SA 4.0'])
      expect(html).toContain(text);
    for (const anchor of ['1-из-каких-подзадач-складывается-ответ', '2-четыре-архитектурных-подхода',
      '3-как-проходит-один-запрос', '4-почему-разрешение-и-структура-неразделимы',
      '5-таблица-и-график-не-являются-линейным-текстом', '6-grounding-как-язык-доказательства',
      '7-возврат-координат-в-оригинал', '8-метрики-на-конкретных-примерах',
      '9-диагностические-вмешательства', '10-контракт-документной-системы'])
      expect(html).toContain('id="' + anchor + '"');
  });


  it('renders video/audio mechanisms, original diagrams and stable anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/video-audio-omni/index.html'), 'utf8');
    for (const name of ['llava-onevision-modalities.png', 'gens-frame-selection.png', 'mimi-split-rvq.png',
      'imagebind-figure1-capabilities.png', 'qwen25omni-figure3-tmrope.png',
      'qwen25omni-figure2-thinker-talker.png', 'qwen3omni-architecture.png',
      'moshi-rq-transformer.png', 'moshi-duplex-streams.png', 'qwen25omni-block-attention.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    expect(html).toContain('class="katex');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    for (const text of ['words_at', 'semantic_with_flush', 'audio_durations', '12.5', 'CC BY-NC-SA 4.0'])
      expect(html).toContain(text);
    for (const anchor of ['1-от-видеофайла-до-visual-tokens', '1-от-видеофайла-до-визуальных-токенов',
      'xgen-video-temporal-compression', 'gens-question-aware-frame-selection',
      '2-номер-кадра-и-физическое-время', '3-от-waveform-до-признаков-и-дискретных-аудиотокенов',
      '3-от-звуковой-волны-к-признакам-и-аудиотокенам', '4-cascade-и-end-to-end-speech-model',
      '4-каскадная-и-сквозная-обработка-речи', '5-общая-временная-ось-аудио-и-видео',
      '6-thinkertalker-понимание-и-речь', '7-streaming-меняет-причинный-контракт',
      '7-потоковая-обработка-и-прерывание-ответа', '8-что-именно-означает-omni',
      '9-матрица-оценивания', '9-оценивание-видео-и-речи'])
      expect(html).toContain('id="' + anchor + '"');
  });

  it('renders VLM evaluation mechanisms, original figures and preserved route anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/multimodal/evaluation-serving/index.html'), 'utf8');
    for (const name of ['mmvet-integrated-capabilities.png', 'cs231n-lecture16-winoground.png',
      'pope-question-construction.png', 'chair-caption-errors.png', 'molmo-figure11-openness.png',
      'epd-stage-interference.png', 'epd-system-architecture.png'])
      expect(html).toMatch(new RegExp('<img[^>]+src="[^"]*/' + name.replace(/\./g, '\\.') + '"'));
    expect(html).toContain('class="katex');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    for (const text of ['winoground_scores', 'binary_metrics', 'critical_path',
      'processor_key', 'block_key', 'quality_ok'])
      expect(html.includes(text), text).toBe(true);
    expect(/(?:&lt;|&#x3[Cc];|&#60;)AND(?:&gt;|&#x3[Ee];|&#62;|>)/.test(html),
      'AND must be rendered as text, not an HTML element').toBe(true);
    for (const anchor of ['1-карта-способностей-и-тестов', '2-как-winoground-изолирует-композиционность',
      '2-как-winoground-проверяет-соответствие-пар', '3-pope-когда-модель-видит-объект-которого-нет',
      '4-проверка-причинной-опоры-на-изображение', '4-проверка-зависимости-ответа-от-изображения',
      '5-открытые-ответы-и-llm-as-a-judge', '6-контаминация-и-открытость',
      '7-serving-полный-путь-одного-запроса', '8-из-чего-состоит-ttft',
      '9-head-of-line-blocking-и-batching', '10-кэш-и-наблюдаемость', '11-безопасность-и-отказ'])
      expect(html).toContain('id="' + anchor + '"');
  });

  it('ships responsive rules for figures, formulas, tables, and attributed source blocks', async () => {
    const assetsDir = join(distDir, '_astro');
    const { readdir } = await import('node:fs/promises');
    const styles = (await readdir(assetsDir)).filter((file) => file.endsWith('.css'));
    const css = (await Promise.all(styles.map((file) => readFile(join(assetsDir, file), 'utf8')))).join('\n');

    expect(css).toContain('.sl-markdown-content img');
    expect(css).toContain('max-width:100%');
    expect(css).toContain('.katex-display');
    expect(css).toContain('overflow-x:auto');
    expect(css).toContain('.source-attribution');
    expect(css).toContain('overflow-wrap:anywhere');
  });

  it('emits every manifest route as Russian HTML', async () => {
    const manifest = YAML.parse(await readFile(join(rootDir, 'publishing', 'navigation.yml'), 'utf8'));
    const routes: string[] = manifest.sections.flatMap(
      (section: { pages: Array<{ route: string }> }) => section.pages.map((page) => page.route)
    );

    for (const route of routes) {
      const outputRoute = route.endsWith('/index') ? route.slice(0, -'/index'.length) : route;
      const html = await readFile(join(distDir, outputRoute, 'index.html'), 'utf8');
      expect(html, route).toMatch(/^<!DOCTYPE html><html lang="ru"/);
    }
  });

  it('does not silently emit empty textbook articles', async () => {
    const manifest = YAML.parse(await readFile(join(rootDir, 'publishing', 'navigation.yml'), 'utf8'));
    const routes: string[] = manifest.sections.flatMap(
      (section: { pages: Array<{ route: string }> }) =>
        section.pages.map((page) => page.route).filter((route) => route.startsWith('textbook/'))
    );

    for (const route of routes) {
      const outputRoute = route.endsWith('/index') ? route.slice(0, -'/index'.length) : route;
      const html = await readFile(join(distDir, outputRoute, 'index.html'), 'utf8');
      expect(html, route).not.toContain('<div class="sl-markdown-content"></div>');
    }
  });

  it('renders scientific discovery mechanisms, examples and original source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/scientific-discovery/index.html'), 'utf8');
    for (const figure of ['lasr-overview.svg', 'lasr-evolution.svg', 'lasr-ablations.svg',
      'escher-overview.svg', 'escher-history-prompt.svg', 'escher-bird-concepts.svg',
      'escher-figure2-original.svg', 'm11-p75-pysr-coulomb.png', 'm11-p76-lasr-coulomb.png'])
      expect(html, figure).toContain(figure);
    for (const mechanism of ['fit_inverse_power', 'lasr_score_v3', 'count_pairs',
      'get_confusion_matrix', 'lasr-experiments', 'f5c96df60ebda4dff1e933a54c9f6f8f03677cf3'])
      expect(html, mechanism).toContain(mechanism);
    for (const alias of ['нейронный-proposal-символическая-проверка-and-search',
      'научный-поиск-как-пространство-проверяемых-гипотез', 'escher-visual-concept-library',
      'что-показали-feynman-equations-and-scaling-law-case'])
      expect(html).toContain(`id="${alias}"`);
    expect(html).toContain('/textbook/agents/formal-math/');
    expect(html).toContain('/textbook/agents/security/');
    expect(html).not.toContain('https://github.com/trishullab/lasr');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders distillation mechanisms, actual examples and complete source comparisons', async () => {
    const html=await readFile(join(distDir,'textbook/post-training/reasoning-distillation/index.html'),'utf8');
    for(const figure of ['kim-rush-three-approaches.png','gkd-forward-reverse-kl.svg',
      'sequence-distillation.png','token-distributions.png','gkd-task-comparison.png',
      'm04-p128-s1-data-funnel.png']) expect(html,figure).toContain(figure);
    for(const mechanism of ['seq_full_gradient','seq_mode_gradient','mixed_gradient',
      'div_candidates','gkd_prefix_length','gkd_loss_after','gkd_teacher_before',
      'sequence_distillation_loss','804 745','generalized_jsd_loss'])
      expect(html,mechanism).toContain(mechanism);
    for(const anchor of ['sequence-distillation','selection-bias','distillation-vs-rft',
      'reasoning-recipes','distillation-evaluation','evaluation'])
      expect(html).toContain('id="'+anchor+'"');
    expect(html).toContain('https://arxiv.org/html/2306.13649v3#A1.F16');
    expect(html).toContain('c6eac19f63be27498f2a7c8108d1e8a4dffde23f');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders GRPO mechanisms, executed examples and complete source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/post-training/grpo-deepseek-r1/index.html'), 'utf8');
    for (const figure of ['ppo-framework.png', 'group-advantages.png', 'length-difficulty-bias.png',
      'length-and-accuracy-ablation.png', 'r1-zero-training-dynamics.png', 'figure2-pipeline.png',
      'kimi-partial-rollouts.svg', 'maxrl-weight-functions.svg'])
      expect(html, figure).toContain(figure);
    for (const mechanism of ['group_advantages', 'clipped_token_loss', 'process_token_advantages',
      'microbatch_loss', 'baseline_expectations', 'maxrl_theoretical', 'maxrl_practical',
      'GRPOConfig', 'offpolicy_surrogate', 'offpolicy_gradients',
      'gspo_gradient', 'kimi_gradient', '0.288', '0.16'])
      expect(html, mechanism).toContain(mechanism);
    for (const anchor of ['rlvr-notation', 'grpo-estimator', 'grpo-normalizations',
      'drgrpo-rft-maxrl', 'offpolicy-ratios', 'gspo', 'reasoning-case-studies'])
      expect(html).toContain('id="' + anchor + '"');
    expect(html).toContain('https://arxiv.org/html/2602.02710v1#S5.F2');
    expect(html).not.toContain('https://arxiv.org/html/2602.02710v1#S5.F3');
    expect(html).toContain('CC BY-NC-ND 4.0');
    expect(html).toContain('/textbook/post-training/reasoning-distillation/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders worked tensor foundations, source illustrations and preserved anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-foundations/tensors/index.html'), 'utf8');
    for (const figure of ['mml-vector-geometry.png', 'broadcasting_2.svg',
      'broadcasting_3.svg', 'broadcasting_4.svg', 'np_transposing_reshaping.png',
      'np_reshape.png', 'tensor-memory-layout.svg'])
      expect(html, figure).toContain(figure);
    expect(html).not.toContain('d2l-broadcasting-example.png');
    expect(html).toContain('source_result');
    expect(html).toContain('Jay Alammar');
    for (const mechanism of ['keepdims', 'centered', 'Осей', 'краткие-итоги'])
      expect(html, mechanism).toContain(mechanism);
    expect(html).toContain('id="суммирование-и-усреднение-по-оси"');
    expect(html).toContain('id="что-происходит-с-несколькими-матрицами"');
    expect(html).toContain('id="почему-нельзя-сразу-сделать-reshape-в-форму-голов"');
    for (const example of ['X_batch', 'W_batch', 'Q_one', 'K_batch', 'swapaxes', 'restored'])
      expect(html, example).toContain(example);
    expect(html).toContain('/textbook/ml-foundations/derivatives-gradients/');
    const broadcasting = html.split('id="поэлементные-операции-и-broadcasting"')[1]
      ?.split('id="суммирование-и-усреднение-по-оси"')[0];
    const dotProduct = html.split('id="скалярное-произведение"')[1]
      ?.split('id="умножение-матриц-как-набор-скалярных-произведений"')[0];
    expect(broadcasting).toBeDefined();
    expect(broadcasting).not.toContain('<ol>');
    expect(dotProduct?.match(/<ol>/g)).toHaveLength(1);
    expect(dotProduct?.match(/<li>/g)).toHaveLength(2);
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the complete tensor storage figure and address walkthrough in root and explicit EN fallback', async () => {
    for (const prefix of ['', 'en/']) {
      const html = await readFile(join(distDir, prefix + 'textbook/ml-foundations/tensors/index.html'), 'utf8');
      expect(html).toContain('id="shape-strides-and-shared-storage"');
      expect(html).toContain('tensor-memory-layout.svg');
      expect(html).toContain('layout_c');
      expect(html).toContain('layout_f');
      expect(html).toContain('layout_slice');
      expect(html).toContain('CC BY-NC-SA 4.0');
      expect(html).toContain('frameworks.qmd#L2904');
      expect(html).toContain('not the result of transposing');
      expect(html).not.toContain('class="katex-error');
      if (prefix) expectCanonicalEditionNotice(html);
    }
  });

  it('preserves all 21 Frameworks source images independently of thematic selection', async () => {
    const report=JSON.parse(await readFile(join(import.meta.dirname,
      '../../docs/audit/2026-10-27-frameworks-native-figures.json'),'utf8'));
    const route = 'sources/courses/harvard-ml-systems/vol1/frameworks';
    const html=await readFile(join(distDir,route,'index.html'),'utf8');
    expect(html.match(/<img\b[^>]*\bsrc=/g)).toHaveLength(21);
    for (const row of report.figures) expect(html).toContain(row.asset.split('/').at(-1));
    expectCourseOriginalAlias(await readFile(join(distDir,'en',route,'index.html'),'utf8'), route);
  });

  it('renders derivative reasoning, Jacobians, checks and complete source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-foundations/derivatives-gradients/index.html'), 'utf8');
    for (const figure of ['mml-secant-to-tangent.png', 'google-gradient-descent.png',
      'mml-computation-graph-forward.png', 'mml-matrix-gradient-shape.png',
      'mml-matrix-gradient-flatten.png'])
      expect(html, figure).toContain(figure);
    for (const mechanism of ['numerical_gradient', '1e-18',
      'Неодинаковые множители и векторные состояния'])
      expect(html, mechanism).toContain(mechanism);
    for (const alias of ['краткие-итоги', 'производная-векторной-функции-якобиан',
      'проверка-градиента-по-координатам', 'точка-излома-требует-отдельной-проверки'])
      expect(html).toContain('id="' + alias + '"');
    expect(html).toContain('/textbook/ml-foundations/tensors/');
    expect(html).toContain('id="прямой-и-обратный-проход-линейного-отображения"');
    expect(html).toContain('numeric_linear');
    expect(html).toContain('/textbook/ml-foundations/probability-likelihood-logarithms/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders probability examples, original figures and the NLL bridge', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-foundations/probability-likelihood-logarithms/index.html'), 'utf8');
    for (const figure of ['mml-discrete-continuous-density.png', 'd2l-likelihood-coin.svg',
      'd2l-coin-frequency-convergence.svg']) expect(html).toContain(figure);
    for (const value of ['stable_log_probability', '0.863422', '715', 'априорной',
      'маргинализац', 'формула Байеса'])
      expect(html.toLowerCase(), value).toContain(value.toLowerCase());
    for (const anchor of ['краткие-итоги', 'независимость-нельзя-предполагать-по-привычке',
      'вероятность-последовательности', 'максимум-правдоподобия-монеты'])
      expect(html).toContain('id="' + anchor + '"');
    expect(html).toContain('id="условное-правдоподобие-что-считается-входом"');
    expect(html).toContain('conditional_likelihood');
    expect(html).toContain('conditional_nll');
    expect(html).toContain('/textbook/ml-foundations/derivatives-gradients/');
    expect(html).toContain('/textbook/ml-foundations/loss-functions/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the worked loss derivations, masks and source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-foundations/loss-functions/index.html'), 'utf8');
    for (const figure of ['cs231n-softmax-cross-entropy.png', 'cs231n-linear-decision-boundaries.png',
      'cs231n-pixelspace-original.jpeg', 'google-mse-outliers.png',
      'google-mae-outliers.png', 'd2l-gaussian-densities.svg', 'd2l-softmaxreg.svg',
      'margin.jpg', 'svmvssoftmax.png'])
      expect(html).toContain(figure);
    for (const code of ['def log_softmax', 'token_mean', 'per_example', 'mask.any', '2.04035515280017'])
      expect(html).toContain(code);
    expect(html).toContain('id="полный-шаг-по-параметрам-классификатора"');
    expect(html).toContain('id="hinge-loss-отступ-между-оценками-классов"');
    expect(html).toContain('id="одни-логиты-разные-функции-потерь"');
    for (const code of ['multiclass_hinge', 'diagram_W', 'diagram_scores', 'comparison_ce'])
      expect(html).toContain(code);
    expect(html).toContain('Ehsan Adeli');
    expect(html).not.toContain('Fei-Fei Li, Ranjay Krishna, Danfei Xu');
    expect(html.match(/<pre data-language="python"/g)).toHaveLength(5);
    expect(html).toContain('id="почему-mse-даёт-среднее-а-mae--медиану"');
    for (const code of ['temperatures', 'mse_values', 'mae_values', 'even_sample'])
      expect(html).toContain(code);
    for (const code of ['W_new', 'b_new', 'loss_new', '1.21178042875392'])
      expect(html).toContain(code);
    for (const anchor of ['почему-градиент-softmax-cross-entropy-особенно-прост', 'краткие-итоги',
      'градиент-по-логитам', 'маски-и-усреднение-по-токенам', 'бинарная-cross-entropy'])
      expect(html).toContain('id="' + anchor + '"');
    expect(html).toContain('/textbook/ml-foundations/probability-likelihood-logarithms/');
    expect(html).toContain('/textbook/ml-foundations/train-validation-test/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the reviewed data split chapter with original diagrams and preserved anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-foundations/train-validation-test/index.html'), 'utf8');
    for (const figure of ['train-validation-test.png', 'model-selection-loop.svg', 'google-hashing-split.gif',
      'mooc-cross-validation-train-test.png', 'sklearn-group-kfold.png', 'sklearn-time-series-split.png'])
      expect(html).toContain(figure);
    for (const anchor of ['три-части--три-разных-вопроса', 'цикл-разработки', 'кросс-валидация',
      'вложенная-кросс-валидация', 'почему-случайный-split-подходит-не-всегда', 'data-leakage',
      'каким-должен-быть-test-set', 'llm-одного-held-out-loss-недостаточно', 'краткие-итоги'])
      expect(html).toContain('id="' + anchor + '"');
    for (const mechanism of ['fold_losses', 'split_group', 'Вложенная кросс-валидация', '0.366', '0.00949'])
      expect(html).toContain(mechanism);
    expect(html).toContain('id="отбор-признаков-высокая-accuracy-на-случайных-метках"');
    for (const code of ['SelectKBest', 'HistGradientBoostingClassifier', 'X_train_selected', 'make_pipeline'])
      expect(html).toContain(code);
    expect(html).toContain('common_pitfalls.rst#data-leakage-during-pre-processing');
    expect(html).toContain('/textbook/ml-foundations/loss-functions/');
    expect(html).toContain('/textbook/ml-foundations/generalization-regularization/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the reviewed generalization chapter with derivations and complete source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/ml-foundations/generalization-regularization/index.html'), 'utf8');
    for (const figure of ['mooc-polynomial-fits.svg', 'mooc-underfit-resamples.svg', 'mooc-overfit-resamples.svg',
      'google-underfit-fit-overfit.svg', 'generalization-curves.png', 'cs229-bias-variance-decomposition.png',
      'google-complex-model.svg', 'google-simple-model.svg', 'google-high-lambda.svg', 'google-low-lambda.svg'])
      expect(html).toContain(figure);
    for (const anchor of ['underfitting-fitting-и-overfitting', 'bias-и-variance-как-диагностическая-модель',
      'регуляризация-меняет-предпочтение-между-решениями', 'weight-decay', 'early-stopping', 'dropout',
      'больше-и-лучше-данных', 'оценка-должна-содержать-неопределённость',
      'почему-масштаб-llm-не-отменяет-overfitting', 'краткие-итоги', 'источники'])
      expect(html).toContain('id="' + anchor + '"');
    for (const example of ['Polynomial.fit', 'bootstrap_means', 'empirical_quantile', '0.634624', '0.9998', '1.93'])
      expect(html).toContain(example);
    expect(html).toContain('id="почему-остановка-может-уменьшить-ошибку-на-новых-данных"');
    for (const example of ['early_path', 'best_step', '0.28125', '0.1953125'])
      expect(html).toContain(example);
    expect(html).toContain('/textbook/ml-foundations/train-validation-test/');
    expect(html).toContain('/textbook/classical-ml/predictive-pipeline/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the reviewed linear models with full derivations and original plots', async () => {
    const html = await readFile(join(distDir, 'textbook/classical-ml/linear-models/index.html'), 'utf8');
    for (const figure of ['mooc-residuals.svg', 'mooc-regression-plane.svg', 'mooc-sigmoid.svg',
      'mooc-logistic-surface.svg', 'mooc-multiclass.svg', 'mooc-nonseparable.svg',
      'mooc-ridge-0.svg', 'mooc-ridge-50.svg', 'mooc-ridge-500.svg',
      'mooc-interpolation-overfit.svg', 'sklearn-polynomial-spline-basis.png',
      'sklearn-polynomial-basis.png', 'sklearn-spline-basis.png'])
      expect(html, figure).toContain(figure);
    for (const anchor of ['linear-regression', 'logistic-regression', 'regularization',
      'геометрия-обучения-и-роль-масштаба', 'мотивирующий-пример-цена-квартиры',
      'нелинейность-через-подготовку-признаков', 'от-оценки-к-классификационному-решению',
      'l1-l2-и-цена-разреженности'])
      expect(html).toContain('id="' + anchor + '"');
    for (const code of ['lstsq', 'gradient_scores', 'loss_after', 'class_probabilities', 'GridSearchCV',
      'minimum_norm', 'residual_sums', 'matrix_rank'])
      expect(html).toContain(code);
    expect(html).toContain('https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html');
    expect(html).toContain('/textbook/classical-ml/predictive-pipeline/');
    expect(html).toContain('/textbook/classical-ml/decision-trees/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    const license = await readFile(join(distDir, 'licenses/scikit-learn-1.9.1-BSD-3-Clause.txt'), 'utf8');
    expect(license).toContain('Copyright (c) 2007-2026 The scikit-learn developers.');
  });

  it('renders clustering mechanisms and real silhouette plots, not a flower', async () => {
    const html = await readFile(join(distDir, 'textbook/classical-ml/clustering/index.html'), 'utf8');
    for (const name of ["elbow.png","silhouette.png","shapes-kmeans.png","shapes-hdbscan.png","dbscan-core-border.png","hdbscan-mst.png","hdbscan-condensed-selected.png","hdbscan-mutual-reachability.svg","lloyd-step-1.svg","lloyd-step-2.svg","lloyd-step-3.svg","lloyd-step-4.svg"]) expect(html, name).toContain(name);
    for (const anchor of ['k-means', 'предположения-k-means', 'как-оценивать-кластеризацию',
      'устойчивость', 'density-based-clustering', 'density-based-методы-на-практике',
      'механика-k-means-как-чередующаяся-оптимизация', 'как-сопоставлять-кластеры-между-запусками',
      'что-считать-успешным-результатом', 'что-нельзя-заключать-по-двумерной-картинке'])
      expect(html).toContain('id="' + anchor + '"');
    for (const code of ['core_sample_indices_', 'n_init', 'cross_val_score']) expect(html).toContain(code);
    for (const explanation of ['Квадраты остаются на прежних местах',
      'не результаты двух', 'Нормировка только наблюдений', 'Weston.pace', 'CC BY-SA 3.0'])
      expect(html).toContain(explanation);
    expect(html).toContain('/textbook/classical-ml/evaluation/');
    expect(html).toContain('/textbook/classical-ml/spectral-clustering/');
    expect(html).not.toContain('iris-silhouette.svg');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
    const license = await readFile(join(distDir, 'licenses/hdbscan-BSD-3-Clause.txt'), 'utf8');
    expect(license).toContain('Copyright (c) 2015, Leland McInnes');
  });

  it('renders evaluation examples, real PR/ROC curves and stable anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/classical-ml/evaluation/index.html'), 'utf8');
    for (const figure of ["pr-curve.png","roc-curve.png","pr-roc.png","calibration-curve.png","calibration-legend.png","calibration.png","regression-actual.png","regression-residual.png","regression-residuals.png","nested-first-outer.png","nested-legend.png","group-kfold.png","time-series-split.png"]) expect(html, figure).toContain(figure);
    for (const anchor of ['cross-validation', 'nested-cross-validation', 'метрики-классификации',
      'score-probability-и-decision', 'calibration', 'regression-metrics',
      'порог-выбирается-по-out-of-fold-предсказаниям', 'минимальный-отчёт',
      'стратификация', 'новые-группы', 'прогноз-будущего'])
      expect(html).toContain('id="' + anchor + '"');
    for (const code of ['TunedThresholdClassifierCV', 'mean_cost', 'average_precision_score'])
      expect(html).toContain(code);
    for (const explanation of ['Зазор можно проследить на 12 объектах',
      'Три сравнения верны, одно нет', 'StratifiedGroupKFold',
      'Он не является способом калибровки или улучшения модели'])
      expect(html).toContain(explanation);
    expect(html).toContain('/textbook/classical-ml/ensembles/');
    expect(html).not.toContain('evaluation_quiz_precision_recall_C1.svg');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the ensemble mechanisms with real residual plots and former anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/classical-ml/ensembles/index.html'), 'utf8');
    for (const name of ["bagging-hundred-average.png","bagging-three-average.png","bagging-three-trees.png","bootstrap-sample.png","ensemble-boosting1.svg","ensemble-boosting2.svg","ensemble-boosting3.svg","gradient-fit-residuals.png","gradient-initial-legend.png","gradient-initial-plot.png","gradient-initial-residuals.png","gradient-residual-legend.png","gradient-residual-plot.png"]) expect(html, name).toContain(name);
    for (const anchor of ['bagging', 'random-forest', 'gradient-boosting', 'histogram-gradient-boosting',
      'bagging-и-boosting--не-одно-и-то-же', 'один-источник-ошибок-две-стратегии',
      'boosting-как-исправление-текущих-ошибок', 'histogram-boosting-и-вычислительный-компромисс',
      'практическая-настройка-и-диагностика'])
      expect(html).toContain('id="' + anchor + '"');
    for (const code of ['estimators_samples_', 'oob_prediction_', 'newton_model', 'stages'])
      expect(html).toContain(code);
    for (const explanation of ['Начальный прогноз тоже выбирают по функции потерь',
      'Начальный логит не обязан равняться нулю', 'До первого дерева все объекты получают вероятность',
      'Голоса с весами', 'В строке AdaBoost приведён разобранный выше бинарный классификатор'])
      expect(html).toContain(explanation);
    expect(html).toContain('/textbook/classical-ml/decision-trees/');
    expect(html).toContain('/textbook/classical-ml/evaluation/');
    expect(html).not.toContain('boosting_iter_orange');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the reviewed decision-tree mechanisms, examples, and source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/classical-ml/decision-trees/index.html'), 'utf8');
    for (const name of ["dt_fit.svg","dt_overfit.svg","dt_underfit.svg","tree2D_1split.svg","tree2D_2split.svg","tree2D_3split.svg","tree_example.svg","tree-example-tight.svg","tree_regression1.svg","tree_regression2.svg","tree_regression4.svg","penguins-tree.png","penguins-probabilities.png","permutation-predictive-feature.png"])
      expect(html, name).toContain(name);
    for (const anchor of ['classification-tree', 'regression-tree', 'размер-дерева-и-generalization',
      'как-дерево-принимает-решение-один-узел-вручную', 'почему-глубокое-дерево-запоминает-данные',
      'рабочий-конвейер', 'интерпретация-без-самообмана'])
      expect(html).toContain('id="' + anchor + '"');
    for (const code of ['cost_complexity_pruning_path', 'ccp_alphas', 'min_impurity_decrease', 'predict_proba',
      'duplicate_stump', 'X_duplicate', 'y_check'])
      expect(html).toContain(code);
    expect(html).toContain('/textbook/classical-ml/linear-models/');
    expect(html).toContain('/textbook/classical-ml/ensembles/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders the reviewed predictive pipeline with source diagrams and former anchors', async () => {
    const html = await readFile(join(distDir, 'textbook/classical-ml/predictive-pipeline/index.html'), 'utf8');
    for (const figure of ['mooc-predictor-fit.svg', 'mooc-predictor-predict.svg', 'mooc-predictor-score.svg',
      'mooc-transform.svg', 'mooc-columns.svg', 'mooc-pipeline-fit.svg', 'mooc-pipeline-predict.svg'])
      expect(html, figure).toContain(figure);
    for (const anchor of ['baseline-задаёт-смысл-метрике', 'train-validation-и-test--разные-полномочия',
      'полный-цикл', 'сквозной-пример-кому-позвонить-сегодня', 'смешанные-типы-данных-без-утечки',
      'почему-предварительная-обработка-входит-в-модель', 'от-функции-потерь-к-качеству-решения'])
      expect(html).toContain('id="' + anchor + '"');
    expect(html).toContain('mean_before');
    expect(html).toContain('sparse_output');
    expect(html).toContain('id="от-обученных-коэффициентов-к-вероятности"');
    for (const example of ['logits_from_parts', 'positive_probability', 'positive_column', '0.662584'])
      expect(html).toContain(example);
    expect(html).toContain('/textbook/classical-ml/linear-models/');
    expect(html).toContain('/textbook/ml-foundations/generalization-regularization/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders executable tokenization examples and the limits of reversibility', async () => {
    const html = await readFile(join(distDir, 'textbook/text-representation/tokenization/index.html'), 'utf8');
    for (const figure of ['tokenized-example.png', 'hf-tokenization-pipeline.svg', 'tokenizer-choice-figure1.png'])
      expect(html, figure).toContain(figure);
    for (const mechanism of ['train_bpe', 'unicodedata', 'byte_fallback', '109/53', '48/109',
      'e5c5054474f583d6d9499624649353995d63c70a', 'Как Unigram обучает вероятности'])
      expect(html, mechanism).toContain(mechanism);
    for (const alias of ['tokenizer-contract', 'bpe-merge-trace', 'tokenizer-evidence'])
      expect(html).toContain(`id="${alias}"`);
    for (const example of ['Почему недостаточно искать самый длинный токен',
      'a · b · cd', 'a · ##bc', 'eedf5716ce1268e56f0a50264a88cafad334ac61',
      '524 288 000', '750 MiB', '9 437 184', 'weight tying'])
      expect(html, example).toContain(example);
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders word-vector objectives, shared-table gradients and all original figures', async () => {
    const html = await readFile(join(distDir, 'textbook/text-representation/word2vec-glove/index.html'), 'utf8');
    for (const figure of ['count-based-idea.png', 'cbow-skipgram.png', 'negative-sampling.png',
      'd2l-hi-softmax.svg', 'glove-idea.png', 'glove-loss.png',
      'bert-feature-extraction-contextualized-embeddings.png', 'gpt2-output-projection-2.png'])
      expect(html, figure).toContain(figure);
    for (const mechanism of ['input_table', 'output_table', 'separate_loss',
      '1.175168994008524', 'getWordVector', '5b5943c118b0ec5fb9cd8d20587de2b2d3966dfe'])
      expect(html, mechanism).toContain(mechanism);
    for (const anchor of ['краткие-итоги', 'два-вклада-в-градиент-общей-таблицы'])
      expect(html).toContain(`id="${anchor}"`);
    expect(html).toContain('/textbook/text-representation/tokenization/');
    expect(html).toContain('/textbook/language-modeling/next-token-probability/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders agent-security mechanisms, version distinctions and source figures', async () => {
    const html = await readFile(join(distDir, 'textbook/agents/security/index.html'), 'utf8');
    for (const figure of ['m12-p13.png', 'm12-p27.png', 'm12-p48.png',
      'agentpoison-overview.png', 'm12-p65.png',
      '12-dawn-song-privilege-control-sequence.png', 'm12-p90.png',
      'datasentinel-detection.png', 'datasentinel-training.svg'])
      expect(html, figure).toContain(figure);
    for (const mechanism of ['sqlite3', 'decide', 'frozenset',
      '2504.11703v1', '2504.11703v3', 'Algorithm 4', 'ASR-r', 'TOCTOU', 'FNR 0,87'])
      expect(html, mechanism).toContain(mechanism);
    for (const alias of ['safety-security-and-the-hybrid-system-boundary',
      'threat-model-before-defenses', 'direct-indirect-and-retrieval-memory-poisoning',
      'progent-least-privilege-as-an-executable-tool-call-policy',
      'monitoring-information-flow-datasentinel-and-formal-policy-verification'])
      expect(html).toContain(`id="${alias}"`);
    expect(html).toContain('/textbook/agents/tool-use/');
    expect(html).toContain('/textbook/agents/scientific-discovery/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('renders TTC selection, upstream search details, cost derivation and source figures', async () => {
    const html = await readFile(join(distDir,'textbook/reasoning/test-time-compute/index.html'),'utf8');
    for (const figure of ['snell-prm-search-methods.png','opro-loop.png','self-discover-stages.png',
      'self-consistency.png','m01-p42-sample-scaling.png','usc-workflow.png',
      'snell-revisions-parallel.png','s1-budget-forcing.png','tot-search.png','tot-game24.png',
      'snell-budget-difficulty.png','snell-sequential-parallel-ratio.png'])
      expect(html,figure).toContain(figure);
    for (const mechanism of ['weighted_vote','expand_frontier','inference_multiplier',
      'get_value','n_generate_sample','21.001','Universal Self-Consistency',
      '8050e67d0e3a0fddc424d7fa5801538722a4c4cc','38%'])
      expect(html,mechanism).toContain(mechanism);
    for (const anchor of ['декомпозиция-и-выбор-reasoning-модулей',
      'budget-forcing-заставить-траекторию-продолжиться','поиск-по-промежуточным-состояниям'])
      expect(html).toContain(`id="${anchor}"`);
    expect(html).toContain('/textbook/post-training/reasoning-distillation/');
    expect(html).not.toContain('class="katex-error');
    expect(html).not.toContain('<p>![');
  });

  it('does not swallow textbook prose into malformed math blocks', async () => {
    const manifest = YAML.parse(await readFile(join(rootDir, 'publishing', 'navigation.yml'), 'utf8'));
    const routes: string[] = manifest.sections.flatMap(
      (section: { pages: Array<{ route: string }> }) =>
        section.pages.map((page) => page.route).filter((route) => route.startsWith('textbook/'))
    );
    const broken: string[] = [];
    for (const route of routes) {
      const outputRoute = route.endsWith('/index') ? route.slice(0, -'/index'.length) : route;
      for (const locale of ['', 'en']) {
        const html = await readFile(join(distDir, locale, outputRoute, 'index.html'), 'utf8');
        if (html.includes('class="katex-error"')) broken.push(`${locale}/${outputRoute}`);
      }
    }
    expect(broken).toEqual([]);
  });

  it('includes previous and next links on an interior handbook page', async () => {
    const html = await readFile(
      join(distDir, 'textbook', 'transformer', 'self-attention', 'index.html'),
      'utf8'
    );
    expect(html).toContain('rel="prev"');
    expect(html).toContain('rel="next"');
  });

  it('retains accessible formulas but excludes duplicate MathML and TeX from search', async () => {
    for (const locale of ['', 'en/']) {
      const html = await readFile(join(distDir, locale, 'textbook/models/gpt-3/index.html'), 'utf8');
      const mathml = [...html.matchAll(/<span[^>]*class="katex-mathml"[^>]*>/g)];
      expect(mathml.length).toBeGreaterThan(10);
      for (const span of mathml) expect(span[0]).toContain('data-pagefind-ignore="all"');
      expect(html).toContain('<annotation encoding="application/x-tex">');
      expect(html).toContain('class="katex-html"');
    }
  });

  it('delivers complete lecture PDFs from official origins without removing their course pages', async () => {
    const cases = [
      ['stanford-cs336-spring-2026', 'Lectures/repository/lecture_09.pdf',
        'https://raw.githubusercontent.com/stanford-cs336/lectures/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf'],
      ['berkeley-advanced-llm-agents-spring-2025', 'Lectures/meeting-04-slides.pdf',
        'https://rdi.berkeley.edu/adv-llm-agents/slides/OLMo-Tulu-Reasoning-Hanna.pdf']
    ];
    for (const [course, pdf, official] of cases) {
      for (const locale of ['', 'en/']) {
        const html = await readFile(join(distDir, locale, 'sources/courses', course, 'index.html'), 'utf8');
        expect(html).toContain(`href="${official}"`);
      }
      await expect(access(join(distDir, 'sources/courses', course, pdf))).rejects.toMatchObject({ code: 'ENOENT' });
      await expect(access(join(rootDir, '05 Источники/Courses', course === 'stanford-cs336-spring-2026'
        ? 'Stanford CS336 Spring 2026' : 'Berkeley Advanced LLM Agents Spring 2025', pdf))).resolves.toBeUndefined();
    }
  });

  it('keeps the full course original and an EN alias that retains old section addresses', async () => {
    const route = 'sources/courses/harvard-ml-systems/tinytorch/19-benchmarking';
    const original = await readFile(join(distDir, route, 'index.html'), 'utf8');
    const alias = await readFile(join(distDir, 'en', route, 'index.html'), 'utf8');
    expect(original).toContain('# jupyter:');
    expect(original).toContain('id="complete-python-source"');
    expect(alias).toContain(`0;url=/${route}/`);
    expect(alias).toContain(`href="/${route}/"`);
    expect(alias).toContain('id="complete-python-source"');
    expect(alias).toContain('location.search + location.hash');
    expect(alias).toContain('noindex');
    expect(alias).not.toContain('# jupyter:');
  });

  it('writes a Russian Pagefind index', async () => {
    await expect(access(join(pagefindDir, 'pagefind.js'))).resolves.toBeUndefined();
    const entry = await readFile(join(pagefindDir, 'pagefind-entry.json'), 'utf8');
    expect(entry).toContain('ru');
  });

  it('finds the attention chapter using its indexed Transformer terminology', async () => {
    const search = await pagefind.search('Transformer');
    const results = await Promise.all(search.results.map((result) => result.data()));
    expect(results.some((result) => result.url.includes('/textbook/transformer/self-attention/')))
      .toBe(true);
  });

  it('stems Russian inflections in the generated index', async () => {
    const search = await pagefind.search('нейронов');
    const results = await Promise.all(search.results.map((result) => result.data()));
    expect(results.some((result) => result.url.includes('/textbook/neural-networks/backpropagation/')))
      .toBe(true);
  });
});
