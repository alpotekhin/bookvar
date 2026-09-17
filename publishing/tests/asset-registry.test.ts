import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { extname, resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
const systemsAssetRoot = resolve(
  root,
  '00 Учебник/Assets/Figures/curated/ml-systems'
);
const trackedFigurePattern = /\.(svg|png|jpe?g|gif|webp)$/i;

type SystemsAsset = {
  file?: unknown;
  author?: unknown;
  source_url?: unknown;
  commit?: unknown;
  license?: unknown;
  modified?: unknown;
  used_in?: unknown;
};

type CuratedAsset = {
  id?: string;
  author?: string;
  license?: string;
  license_url?: string;
  asset: string;
  derivation?: string;
  provenance_confirmation?: string;
  source_asset?: string;
  source_asset_sha256?: string;
  source_page?: number;
  source_url?: string;
  commit?: string;
  metadata_status?: string;
  modifications?: string;
  sha256?: string;
};

function requireNonEmptyString(value: unknown, field: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`${field} must be a non-empty string`);
  }
  return value;
}

function validateSystemsManifest(assets: SystemsAsset[]): void {
  const files = new Set<string>();

  for (const asset of assets) {
    const file = requireNonEmptyString(asset.file, 'file');
    requireNonEmptyString(asset.author, `author for ${file}`);
    requireNonEmptyString(asset.license, `license for ${file}`);
    if (typeof asset.modified !== 'boolean') {
      throw new Error(`modified for ${file} must be boolean`);
    }
    if (!Array.isArray(asset.used_in) || asset.used_in.length === 0) {
      throw new Error(`used_in for ${file} must be a non-empty array`);
    }
    for (const usage of asset.used_in) {
      requireNonEmptyString(usage, `used_in item for ${file}`);
    }

    const sourceUrl = requireNonEmptyString(asset.source_url, `source_url for ${file}`);
    const sourceMatch = sourceUrl.match(/^https:\/\/github\.com\/.+\/blob\/([0-9a-f]{40})\//);
    if (!sourceMatch) {
      throw new Error(`source_url for ${file} must pin a 40-character GitHub commit`);
    }
    const commit = requireNonEmptyString(asset.commit, `commit for ${file}`);
    if (!/^[0-9a-f]{40}$/.test(commit)) {
      throw new Error(`commit for ${file} must be a 40-character SHA`);
    }
    if (sourceMatch[1] !== commit) {
      throw new Error(`source_url SHA must equal commit for ${file}`);
    }
    if (files.has(file)) {
      throw new Error(`duplicate manifest file: ${file}`);
    }
    files.add(file);
  }
}

function sha256(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function validateCuratedSourceAsset(entry: CuratedAsset, repositoryRoot = root): void {
  if (!entry.source_asset) {
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    const sourceMatch = entry.source_url?.match(
      /^https:\/\/github\.com\/[^/]+\/[^/]+\/blob\/([0-9a-f]{40})\/.+/
    );
    expect(sourceMatch, `Curated upstream source must be a pinned GitHub blob: ${entry.asset}`).toBeTruthy();
    expect(entry.commit).toMatch(/^[0-9a-f]{40}$/);
    expect(sourceMatch?.[1]).toBe(entry.commit);
    expect(entry.metadata_status).toBe('verified');
    return;
  }

  if (entry.provenance_confirmation === 'source-license-verified') {
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.license, 'license');
    expect(entry.source_url).toMatch(/^https:\/\//);
    expect(entry.license_url).toMatch(/^https:\/\//);
    expect(entry.metadata_status).toBe('verified');
  } else {
    expect(entry.provenance_confirmation).toBe('user-confirmed-open-materials');
  }
  if (entry.source_asset.startsWith('raw/papers/')) {
    const sourceAsset = resolve(repositoryRoot, entry.source_asset);
    if (!existsSync(sourceAsset)) {
      // `raw` is intentionally ignored in the public repository. CI validates
      // the curated binary and its provenance metadata; a local checkout that
      // has the immutable source archive also verifies byte-level derivation.
      return;
    }
    if (entry.derivation === 'pdf-page-render-crop') {
      expect(entry.source_asset).toMatch(/\.pdf$/);
    } else if (entry.derivation === 'lossless-format-conversion') {
      expect(entry.source_asset).toMatch(/\.jpe?g$/i);
      expect(entry.asset).toMatch(/\.png$/i);
      expect(entry.modifications).toMatch(/converted to PNG without resizing or content changes/i);
    } else {
      expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(sha256(sourceAsset));
    }
    return;
  }

  expect(entry.source_asset).toMatch(
    /^05 Источники\/Courses\/[^/]+\/Lectures\/[^/]+\.pdf$/
  );
  expect(entry.derivation).toBe('pdf-page-render');
  expect(entry.source_asset_sha256).toMatch(/^[a-f0-9]{64}$/);
  expect(Number.isInteger(entry.source_page) && Number(entry.source_page) > 0).toBe(true);
  const sourceAsset = resolve(repositoryRoot, entry.source_asset);
  expect(existsSync(sourceAsset), `Missing tracked course PDF: ${entry.source_asset}`).toBe(true);
  expect(sha256(sourceAsset), `Source PDF hash mismatch: ${entry.source_asset}`).toBe(
    entry.source_asset_sha256
  );
  expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
  expect(sha256(resolve(repositoryRoot, entry.asset)), `Curated output hash mismatch: ${entry.asset}`).toBe(
    entry.sha256
  );
}

function expectDecodableImage(path: string): void {
  const bytes = readFileSync(path);
  const extension = extname(path).toLowerCase();
  const prefix = Buffer.from(bytes.subarray(0, 32)).toString('utf8').trimStart().toLowerCase();
  expect(prefix, `${path} contains HTML instead of image data`).not.toMatch(/^<!doctype html|^<html/);

  if (extension === '.png') {
    expect([...bytes.subarray(0, 8)], `${path} has an invalid PNG signature`).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
    expect(Buffer.from(bytes).includes(Buffer.from('IEND')), `${path} has no PNG IEND chunk`).toBe(true);
  } else if (extension === '.jpg' || extension === '.jpeg') {
    expect([...bytes.subarray(0, 3)], `${path} has an invalid JPEG signature`).toEqual([255, 216, 255]);
    expect([...bytes.subarray(-2)], `${path} has no JPEG end marker`).toEqual([255, 217]);
  } else if (extension === '.gif') {
    expect(Buffer.from(bytes.subarray(0, 6)).toString('ascii'), `${path} has an invalid GIF signature`).toMatch(/^GIF8[79]a$/);
    expect(bytes.at(-1), `${path} has no GIF trailer`).toBe(0x3b);
  } else if (extension === '.webp') {
    expect(Buffer.from(bytes.subarray(0, 4)).toString('ascii'), `${path} has an invalid RIFF signature`).toBe('RIFF');
    expect(Buffer.from(bytes.subarray(8, 12)).toString('ascii'), `${path} has an invalid WebP signature`).toBe('WEBP');
  } else if (extension === '.svg') {
    const text = readFileSync(path, 'utf8').replace(/^\uFEFF/, '').trimStart();
    expect(text, `${path} has no SVG root`).toMatch(
      /^(?:<\?xml[^>]*>\s*)?(?:<!DOCTYPE\s+svg[^>]*(?:\[[^]*?\]\s*)?>\s*)?(?:<!--[^]*?-->\s*)*<svg[\s>]/i
    );
    expect(text, `${path} contains HTML instead of SVG`).not.toMatch(/<!doctype html|<html[\s>]/i);
  } else {
    throw new Error(`Unsupported tracked figure extension: ${path}`);
  }
}

describe('publication asset registry', () => {
  it('keeps the original Kimi Linear JPEG bytes and its complete MIT notice', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entry = registry.assets.find((asset) => asset.id === 'curated-kimi-linear-architecture-132ae021fa46')!;
    expect(entry).toBeTruthy();
    expect(entry.asset).toMatch(/arch\.jpg$/);
    validateCuratedSourceAsset(entry);
    expect(sha256(resolve(root, entry.asset))).toBe('132ae021fa4661ed39e7be784d46f05f22b82aabb9afd2bab8dbdc0a5a61cba0');
    expectDecodableImage(resolve(root, entry.asset));
    const notice = readFileSync(resolve(root, 'site/public/licenses/kimi-linear-MIT.txt'), 'utf8');
    expect(notice).toContain('Copyright (c) 2025 Moonshot AI');
    expect(notice).toContain('The above copyright notice and this permission notice shall be included');
    expect(notice).toContain('THE SOFTWARE IS PROVIDED "AS IS"');
  });

  it('attributes the three speculative-sampling figures to Chen with the published license', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const ids = ['curated-spec-alg-000000000047', 'curated-spec-results-000000000048', 'curated-spec-stats-000000000049'];
    for (const id of ids) {
      const entry = registry.assets.find((asset) => asset.id === id);
      expect(entry).toBeTruthy();
      if (!entry) continue;
      expect(entry.author).toBe('Charlie Chen et al.');
      expect(entry.source_url).toBe('https://arxiv.org/abs/2302.01318');
      expect(entry.license).toBe('CC BY 4.0');
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
      validateCuratedSourceAsset(entry);
      expect(() => validateCuratedSourceAsset({ ...entry, license: '' })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, license_url: undefined })).toThrow();
    }
  });

  it('registers every curated ML systems asset with pinned provenance', () => {
    const manifestPath = resolve(systemsAssetRoot, 'assets.yml');
    const manifest = parse(readFileSync(manifestPath, 'utf8')) as {
      assets: Array<{
        file: string;
        author: string;
        source_url: string;
        commit: string;
        license: string;
        modified: boolean;
        used_in: string[];
      }>;
    };
    const registered = new Set(manifest.assets.map((asset) => asset.file));
    const actual = readdirSync(systemsAssetRoot, { recursive: true })
      .map(String)
      .filter((file) => /\.(svg|png|jpe?g|gif|webp)$/i.test(file));

    validateSystemsManifest(manifest.assets);
    expect([...registered].sort()).toEqual(actual.sort());
  });

  const commit = '0123456789abcdef0123456789abcdef01234567';
  const validSystemsAsset = {
    file: 'figure.svg',
    author: 'Example Author',
    source_url: `https://github.com/example/course/blob/${commit}/figure.svg`,
    commit,
    license: 'CC BY 4.0',
    modified: false,
    used_in: ['00 Учебник/example.md']
  };
  const invalidSystemsManifests: Array<[string, SystemsAsset[], RegExp]> = [
    ['empty file', [{ ...validSystemsAsset, file: '  ' }], /file/],
    ['empty author', [{ ...validSystemsAsset, author: '' }], /author/],
    ['empty license', [{ ...validSystemsAsset, license: '' }], /license/],
    ['non-boolean modified', [{ ...validSystemsAsset, modified: 'false' }], /modified/],
    ['empty used_in', [{ ...validSystemsAsset, used_in: [] }], /used_in/],
    ['blank used_in item', [{ ...validSystemsAsset, used_in: [''] }], /used_in/],
    ['unpinned source_url', [{ ...validSystemsAsset, source_url: 'https://github.com/example/course/blob/main/figure.svg' }], /source_url/],
    ['invalid commit', [{ ...validSystemsAsset, commit: 'main' }], /commit/],
    ['duplicate file', [validSystemsAsset, { ...validSystemsAsset }], /duplicate.*file/i],
    ['source SHA mismatch', [{ ...validSystemsAsset, commit: 'abcdef0123456789abcdef0123456789abcdef01' }], /source_url.*commit/i]
  ];

  it.each(invalidSystemsManifests)('rejects ML systems manifest rows with %s', (_label, assets, error) => {
    expect(() => validateSystemsManifest(assets)).toThrow(error);
  });

  it('stores decodable image data matching every tracked figure extension', () => {
    const figuresRoot = resolve(root, '00 Учебник/Assets/Figures');
    const files = readdirSync(figuresRoot, { recursive: true, withFileTypes: true })
      .filter((entry) => entry.isFile() && trackedFigurePattern.test(entry.name))
      .map((entry) => `${entry.parentPath}/${entry.name}`);

    for (const file of files) expectDecodableImage(file);
  });

  it('records every tracked textbook figure with reviewable rights metadata', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: Array<Record<string, unknown>>;
    };
    const paths = registry.assets.map((asset) => asset.asset);
    const figuresRoot = resolve(root, '00 Учебник/Assets/Figures');
    const files = readdirSync(figuresRoot, { recursive: true, withFileTypes: true })
      .filter((entry) => entry.isFile() && trackedFigurePattern.test(entry.name))
      .map((entry) => `00 Учебник/Assets/Figures/${entry.parentPath.slice(figuresRoot.length + 1)}${entry.parentPath === figuresRoot ? '' : '/'}${entry.name}`)
      .sort();

    expect(paths).toHaveLength(new Set(paths).size);
    expect(paths.toSorted()).toEqual(files);
    expect(paths).toContain('00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png');
    for (const asset of registry.assets) {
      expect(typeof asset.asset).toBe('string');
      expect(asset.author === null || typeof asset.author === 'string').toBe(true);
      expect(asset.source_url === null || typeof asset.source_url === 'string').toBe(true);
      expect(asset.license === null || typeof asset.license === 'string').toBe(true);
      expect(asset.license_url === null || typeof asset.license_url === 'string').toBe(true);
      expect(typeof asset.modifications).toBe('string');
      expect(['verified', 'needs-confirmation']).toContain(asset.metadata_status);
      expect(Array.isArray(asset.used_in)).toBe(true);
      for (const usage of asset.used_in as unknown[]) {
        expect(typeof usage).toBe('string');
        expect(existsSync(resolve(root, usage as string)), `Missing used_in note: ${String(usage)}`).toBe(true);
      }

      if (asset.metadata_status === 'verified') {
        expect(typeof asset.author).toBe('string');
        expect(typeof asset.source_url).toBe('string');
        expect(typeof asset.license).toBe('string');
        expect(typeof asset.license_url).toBe('string');
      }
    }

    const attention = registry.assets.find((asset) =>
      asset.asset === '00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png'
    );
    expect(attention?.used_in).toEqual([
      '00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md',
      'Concepts/Architectures/Transformer.md',
      'Concepts/NLP/Attention Mechanism.md'
    ]);

    const deepseekR1 = registry.assets.find((asset) =>
      asset.asset === '00 Учебник/Assets/Figures/deepseek-r1-figure1-hq.png'
    );
    expect(deepseekR1?.used_in).toEqual([
      '00 Учебник/10 Атлас современных архитектур/01 Llama, Qwen и DeepSeek как эволюция блока.md',
      '00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1.md',
      'Papers/DeepSeek-R1 Reasoning via RL.md'
    ]);
  });

  it('publishes without retaining any raw-asset-blocked allowlist entry', () => {
    const allowlist = JSON.parse(readFileSync(resolve(root, 'publishing/link-allowlist.json'), 'utf8')) as {
      entries: Array<{ target: string; reason: string }>;
    };
    const blocked = allowlist.entries.filter(({ reason }) =>
      reason === 'authored page depends on ignored raw assets; publish after asset curation'
    );
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: Array<{ asset: string }>;
    };
    const registeredAssets = new Set(registry.assets.map(({ asset }) => asset));
    const noteRoots = ['00 Учебник', 'Concepts', 'Papers'];
    const authoredNotes = noteRoots.flatMap((directory) => {
      const directoryRoot = resolve(root, directory);
      return readdirSync(directoryRoot, { recursive: true, withFileTypes: true })
        .filter((entry) => entry.isFile() && entry.name.endsWith('.md'))
        .map((entry) => `${entry.parentPath}/${entry.name}`);
    });

    expect(blocked).toHaveLength(0);
    for (const { target } of blocked) {
      const relativePage = target.replace(/^02 Areas\/ML & DL\//, '');
      const exactPage = resolve(root, `${relativePage}.md`);
      const basenameMatches = relativePage.includes('/')
        ? []
        : authoredNotes.filter((page) => page.endsWith(`/${relativePage}.md`));
      const page = existsSync(exactPage) ? exactPage : basenameMatches[0] ?? exactPage;
      expect(existsSync(exactPage) || basenameMatches.length === 1, `Ambiguous curated target: ${relativePage}`).toBe(true);
      expect(existsSync(page), `Missing curated target page: ${relativePage}`).toBe(true);
      const source = readFileSync(page, 'utf8');
      expect(source, `Raw asset remains in ${relativePage}`).not.toMatch(
        /!\[\[[^\]]*(?:^|\/)raw\/|!\[[^\]]*\]\([^)]*(?:^|\/)raw\//m
      );
      const localImages = [
        ...source.matchAll(/!\[\[((?:02 Areas\/ML & DL\/)?[^\]|]+\.(?:png|jpe?g|gif|webp|svg))(?:\|[^\]]*)?\]\]/giu),
        ...source.matchAll(/!\[[^\]]*\]\(((?:02 Areas\/ML & DL\/)?[^)]+\.(?:png|jpe?g|gif|webp|svg))\)/giu)
      ].map((match) => match[1]?.replace(/^02 Areas\/ML & DL\//, ''));
      for (const image of localImages) {
        expect(image, `Invalid local image in ${relativePage}`).toBeTypeOf('string');
        expect(registeredAssets.has(image as string), `Unregistered local image in ${relativePage}: ${image}`).toBe(true);
        expect(existsSync(resolve(root, image as string)), `Missing tracked image in ${relativePage}: ${image}`).toBe(true);
      }
    }

    for (const page of authoredNotes) {
      const source = readFileSync(page, 'utf8');
      expect(source, `Raw asset remains in publication candidate: ${page}`).not.toMatch(
        /!\[\[[^\]]*(?:^|\/)raw\/|!\[[^\]]*\]\([^)]*(?:^|\/)raw\//m
      );
    }
  });

  it('registers every tracked curated binary exactly once', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: CuratedAsset[];
    };
    const curatedRoot = resolve(root, '00 Учебник/Assets/Figures/curated');
    const files = existsSync(curatedRoot)
      ? readdirSync(curatedRoot, { recursive: true, withFileTypes: true })
        .filter((entry) => entry.isFile() && trackedFigurePattern.test(entry.name))
        .map((entry) => `00 Учебник/Assets/Figures/curated/${entry.parentPath.slice(curatedRoot.length + 1)}${entry.parentPath === curatedRoot ? '' : '/'}${entry.name}`)
        .sort()
      : [];
    const registered = registry.assets
      .map(({ asset }) => asset)
      .filter((asset) => asset.startsWith('00 Учебник/Assets/Figures/curated/'))
      .sort();

    expect(files.length).toBeGreaterThan(0);
    expect(registered).toEqual(files);
    const curatedEntries = registry.assets.filter(({ asset }) =>
      asset.startsWith('00 Учебник/Assets/Figures/curated/')
    );
    expect(new Set(curatedEntries.map(({ id }) => id)).size).toBe(curatedEntries.length);
    for (const entry of curatedEntries) {
      expect(entry.id).toMatch(/^curated-[a-z0-9-]+-[a-f0-9]{12}$/);
      validateCuratedSourceAsset(entry);
    }
  });

  it('rejects invalid tracked course PDF provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: CuratedAsset[];
    };
    const valid = registry.assets.find(({ asset }) =>
      asset.startsWith('00 Учебник/Assets/Figures/curated/berkeley-agents-2025/')
    );
    expect(valid).toBeTruthy();
    if (!valid) return;

    expect(() => validateCuratedSourceAsset({ ...valid, source_asset: 'elsewhere/course.pdf' })).toThrow();
    expect(() => validateCuratedSourceAsset({
      ...valid,
      source_asset: '05 Источники/Courses/missing/Lectures/missing.pdf'
    })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...valid, source_asset_sha256: '0'.repeat(64) })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...valid, source_page: 0 })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...valid, derivation: 'pdf-page-render-crop' })).toThrow();
  });
});
