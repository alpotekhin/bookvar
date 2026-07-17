import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { extname, resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');

function sha256(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
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
    expect(text, `${path} has no SVG root`).toMatch(/^(?:<\?xml[^>]*>\s*)?(?:<!--[^]*?-->\s*)*<svg[\s>]/i);
    expect(text, `${path} contains HTML instead of SVG`).not.toMatch(/<!doctype html|<html[\s>]/i);
  } else {
    throw new Error(`Unsupported tracked figure extension: ${path}`);
  }
}

describe('publication asset registry', () => {
  it('stores decodable image data matching every tracked figure extension', () => {
    const figuresRoot = resolve(root, '00 Учебник/Assets/Figures');
    const files = readdirSync(figuresRoot, { recursive: true, withFileTypes: true })
      .filter((entry) => entry.isFile())
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
      .filter((entry) => entry.isFile())
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
      assets: Array<{
        id?: string;
        asset: string;
        derivation?: string;
        provenance_confirmation?: string;
        source_asset?: string;
      }>;
    };
    const curatedRoot = resolve(root, '00 Учебник/Assets/Figures/curated');
    const files = existsSync(curatedRoot)
      ? readdirSync(curatedRoot, { recursive: true, withFileTypes: true })
        .filter((entry) => entry.isFile())
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
      expect(entry.provenance_confirmation).toBe('user-confirmed-open-materials');
      expect(entry.source_asset).toMatch(/^raw\/papers\//);
      if (entry.derivation === 'pdf-page-render-crop') {
        expect(entry.source_asset).toMatch(/\.pdf$/);
        expect(existsSync(resolve(root, entry.source_asset as string))).toBe(true);
      } else {
        expect(sha256(resolve(root, entry.asset))).toBe(
          sha256(resolve(root, entry.source_asset as string))
        );
      }
    }
  });
});
