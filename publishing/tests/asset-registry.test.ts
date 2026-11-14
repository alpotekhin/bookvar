import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { extname, resolve } from 'node:path';
import { deflateSync, inflateSync } from 'node:zlib';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const fullTransformerFigures = JSON.parse(readFileSync(resolve(root,
  'publishing/tools/full-transformer-figures.json'),'utf8'));
const architectureFigures = JSON.parse(readFileSync(resolve(root,
  'publishing/tools/architecture-figures.json'),'utf8'));
const bertFamilyFigures = JSON.parse(readFileSync(resolve(root,
  'publishing/tools/bert-family-figures.json'),'utf8'));
const gpt2Figures = JSON.parse(readFileSync(resolve(root,
  'publishing/tools/gpt2-figures.json'),'utf8'));
const gpt3Figures = JSON.parse(readFileSync(resolve(root,
  'publishing/tools/gpt3-figures.json'),'utf8'));
const retiredFigureCopies = JSON.parse(readFileSync(resolve(root,
  '05 Источники/Originals/figure-replacements-2026-10-30/manifest.json'),'utf8')).retired;
const t5Figures = JSON.parse(readFileSync(resolve(root,
  'publishing/tools/t5-figures.json'),'utf8'));
const llamaFigures = JSON.parse(readFileSync(resolve(root,
  'publishing/tools/llama-figures.json'),'utf8'));
const decoderBlockFigures = JSON.parse(readFileSync(resolve(root,
  'publishing/tools/decoder-block-figures.json'),'utf8'));
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
  license_url?: string | null;
  asset: string;
  derivation?: string;
  provenance_confirmation?: string;
  source_asset?: string;
  source_asset_sha256?: string;
  source_archive_member?: string;
  source_archive_sha256?: string;
  source_parent_sha256?: string;
  figure_url?: string;
  publication_status?: string;
  retirement_reason?: string;
  permission_asset?: string;
  permission_asset_sha256?: string;
  source_stylesheet_asset?: string;
  source_stylesheet_url?: string;
  source_stylesheet_sha256?: string;
  license_evidence?: string;
  license_evidence_sha256?: string;
  license_evidence_url?: string;
  source_member?: string;
  source_member_sha256?: string;
  source_cell?: number;
  source_cell_index?: number;
  source_output_index?: number;
  source_page?: number;
  source_pdf_object?: number;
  source_printed_page?: number;
  render_dpi?: number;
  render_scale_to_px?: number;
  crop_box_px?: unknown;
  render_page_size_px?: unknown;
  rights_status?: string;
  source_url?: string;
  source_image_url?: string;
  source_blob_sha1?: string;
  source_code?: string;
  source_code_url?: string;
  source_code_sha256?: string;
  source_page_snapshot?: string;
  source_page_sha256?: string;
  source_viewbox?: number[];
  public_license_copy?: string;
  commit?: string;
  metadata_status?: string;
  modifications?: string;
  sha256?: string;
  dimensions_px?: number[];
  frames?: number;
  bytes?: number;
  used_in?: string[];
  former_used_in?: string[];
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

function sourceSnapshotSha256(repositoryRoot: string, path: string): string {
  const absolute = resolve(repositoryRoot, path);
  if (!/^05 Источники\/Courses\/Harvard ML Systems\/vol1\/(frameworks|nn_architectures)\.md$/.test(path))
    return sha256(absolute);
  const source = readFileSync(absolute, 'utf8');
  const archived = beforeTopicConsolidation(source, path, repositoryRoot);
  return createHash('sha256').update(archived).digest('hex');
}

function expectViewportOnlySvgCrop(original: string, cropped: string): void {
  const pattern = /viewBox="([^"]+)"/g;
  const before = [...original.matchAll(pattern)];
  const after = [...cropped.matchAll(pattern)];
  expect(before).toHaveLength(1);
  expect(after).toHaveLength(1);
  const full = before[0]![1]!.trim().split(/\s+/).map(Number);
  const view = after[0]![1]!.trim().split(/\s+/).map(Number);
  expect(full).toHaveLength(4);
  expect(view).toHaveLength(4);
  expect([...full, ...view].every(Number.isFinite)).toBe(true);
  expect(view[2]).toBeGreaterThan(0);
  expect(view[3]).toBeGreaterThan(0);
  expect(view[0]).toBeGreaterThanOrEqual(full[0]!);
  expect(view[1]).toBeGreaterThanOrEqual(full[1]!);
  expect(view[0]! + view[2]!).toBeLessThanOrEqual(full[0]! + full[2]!);
  expect(view[1]! + view[3]!).toBeLessThanOrEqual(full[1]! + full[3]!);
  expect(cropped).toBe(original.replace(before[0]![0], after[0]![0]));
}

function pngDimensions(path: string): number[] {
  const bytes = readFileSync(path);
  expect(bytes.length, `Truncated PNG: ${path}`).toBeGreaterThanOrEqual(33);
  expect([...bytes.subarray(0, 8)], `Invalid PNG signature: ${path}`).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
  expect(bytes.readUInt32BE(8), `Invalid PNG IHDR length: ${path}`).toBe(13);
  expect(bytes.toString('ascii', 12, 16), `Missing PNG IHDR: ${path}`).toBe('IHDR');
  const dimensions = [bytes.readUInt32BE(16), bytes.readUInt32BE(20)];
  expect(dimensions.every(value => value > 0), `Invalid PNG dimensions: ${path}`).toBe(true);
  return dimensions;
}

// Decode the lossless pixel stream for the new PNG derivations, rather than
// accepting a file that only has an image signature and an IEND substring.
// Accept non-interlaced 8-bit RGB/RGBA, grayscale and original indexed PNGs.
function pngCrc(bytes: Buffer): number {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function validateRenderedPngBytes(bytes: Buffer): number[] {
  expect([...bytes.subarray(0, 8)]).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
  let offset = 8;
  let header: Buffer | undefined;
  let palette: Buffer | undefined;
  let ended = false;
  const compressed: Buffer[] = [];
  while (offset < bytes.length) {
    expect(offset + 12).toBeLessThanOrEqual(bytes.length);
    const length = bytes.readUInt32BE(offset);
    const end = offset + 12 + length;
    expect(end).toBeLessThanOrEqual(bytes.length);
    const kind = bytes.toString('ascii', offset + 4, offset + 8);
    expect(pngCrc(bytes.subarray(offset + 4, end - 4)), `Invalid PNG ${kind} CRC`).toBe(bytes.readUInt32BE(end - 4));
    const data = bytes.subarray(offset + 8, end - 4);
    if (kind === 'IHDR') {
      expect(offset).toBe(8);
      expect(header).toBeUndefined();
      expect(length).toBe(13);
      header = data;
    } else if (kind === 'PLTE') {
      expect(header).toBeDefined();
      expect(palette).toBeUndefined();
      expect(compressed).toHaveLength(0);
      expect(length > 0 && length <= 768 && length % 3 === 0).toBe(true);
      palette = data;
    } else if (kind === 'IDAT') {
      expect(header).toBeDefined();
      compressed.push(data);
    } else if (kind === 'IEND') {
      expect(length).toBe(0);
      expect(end).toBe(bytes.length);
      ended = true;
    }
    offset = end;
  }
  expect(ended).toBe(true);
  expect(header).toBeDefined();
  expect(compressed.length).toBeGreaterThan(0);
  const [width, height] = [header!.readUInt32BE(0), header!.readUInt32BE(4)];
  expect(width).toBeGreaterThan(0);
  expect(height).toBeGreaterThan(0);
  expect(header![8]).toBe(8); // bit depth
  const channels = new Map([[0, 1], [2, 3], [3, 1], [4, 2], [6, 4]]).get(header![9]);
  expect(channels).toBeDefined();
  expect([...header!.subarray(10)]).toEqual([0, 0, 0]); // compression, filter, interlace
  const stride = 1 + width * channels!;
  const decoded = inflateSync(Buffer.concat(compressed), { maxOutputLength: stride * height });
  expect(decoded.length).toBe(stride * height);
  for (let y = 0; y < height; y++) expect(decoded[y * stride]).toBeLessThanOrEqual(4);
  if (header![9] === 3) {
    expect(palette).toBeDefined();
    let previous = Buffer.alloc(width), maxIndex = 0;
    for (let y = 0; y < height; y++) {
      const filter = decoded[y * stride], row = Buffer.alloc(width);
      for (let x = 0; x < width; x++) {
        const left = x ? row[x - 1] : 0, up = previous[x];
        const upperLeft = x ? previous[x - 1] : 0;
        const p = left + up - upperLeft;
        const pa = Math.abs(p - left), pb = Math.abs(p - up), pc = Math.abs(p - upperLeft);
        const paeth = pa <= pb && pa <= pc ? left : pb <= pc ? up : upperLeft;
        const predictor = [0, left, up, Math.floor((left + up) / 2), paeth][filter];
        row[x] = (decoded[y * stride + 1 + x] + predictor) & 255;
        maxIndex = Math.max(maxIndex, row[x]);
      }
      previous = row;
    }
    expect(maxIndex).toBeLessThan(palette!.length / 3);
  }
  return [width, height];
}

const d2lCommit = '23d7a5aecceee57d1292c56e90cce307f183bb0a';
const cs336Commit = '8b59b50730766695c2ffedd1a79c50cd09b9eb91';
const curatedPngPath = /^00 Учебник\/Assets\/Figures\/curated\/(?:[\w-]+\/)+[\w-]+\.png$/;

const wordEmbeddingFigures = {
  "00 Учебник/Assets/Figures/curated/voita-word-embeddings/count-based-idea.png": {
    "hash": "bdd6d2fcd48116d0fe055e6fceaa051a4b95128c4367d5c591102f5ee0dcbd05",
    "member": "voita-count-based-idea.png",
    "url": "https://lena-voita.github.io/resources/lectures/word_emb/preneural/idea-min.png",
    "width": 2508,
    "height": 1070,
    "kind": "voita"
  },
  "00 Учебник/Assets/Figures/curated/voita-word-embeddings/cbow-skipgram.png": {
    "hash": "5b9be7c43242ffc141a99b3f16df429a9c893dcf0c7b85a627f7475ae9b4585f",
    "member": "voita-cbow-skipgram.png",
    "url": "https://lena-voita.github.io/resources/lectures/word_emb/w2v/cbow_skip-min.png",
    "width": 2524,
    "height": 1368,
    "kind": "voita"
  },
  "00 Учебник/Assets/Figures/curated/voita-word-embeddings/negative-sampling.png": {
    "hash": "8563b306dab2b8ea5453dc05f94447ba31d02dfcb71ba16046b07ad39108430c",
    "member": "voita-negative-sampling.png",
    "url": "https://lena-voita.github.io/resources/lectures/word_emb/w2v/negative_sampling-min.png",
    "width": 2325,
    "height": 1159,
    "kind": "voita"
  },
  "00 Учебник/Assets/Figures/curated/voita-word-embeddings/glove-idea.png": {
    "hash": "1d189da12c2023582b347ff563b24d77ee9bc6cb33339f051eb67217bb2bfed3",
    "member": "voita-glove-idea.png",
    "url": "https://lena-voita.github.io/resources/lectures/word_emb/glove/idea-min.png",
    "width": 2342,
    "height": 642,
    "kind": "voita"
  },
  "00 Учебник/Assets/Figures/curated/voita-word-embeddings/glove-loss.png": {
    "hash": "516f291039ad5aedf0a046e6d9eb56464c4c88947c774e54b65a6f8f6a936d5d",
    "member": "voita-glove-loss.png",
    "url": "https://lena-voita.github.io/resources/lectures/word_emb/glove/glove_loss-min.png",
    "width": 2283,
    "height": 1093,
    "kind": "voita"
  },
  "00 Учебник/Assets/Figures/curated/bert/bert-feature-extraction-contextualized-embeddings.png": {
    "hash": "ab1792bd6e14283e015f2bc60647de48df1e11c6fe7b270382800c0ca5865ebc",
    "member": "jay-bert-layers.png",
    "url": "https://jalammar.github.io/images/bert-feature-extraction-contextualized-embeddings.png",
    "width": 1287,
    "height": 776,
    "kind": "jay"
  },
  "00 Учебник/Assets/Figures/curated/word-embeddings-2026-10-21/d2l-hi-softmax.svg": {
    "hash": "2d8c76e194b90130d364694d0429ba84df3927f6c14ba0ebe9d5e442a08a152c",
    "member": "d2l-hi-softmax.svg",
    "url": "https://raw.githubusercontent.com/d2l-ai/d2l-en/23d7a5aecceee57d1292c56e90cce307f183bb0a/img/hi-softmax.svg",
    "kind": "d2l"
  }
} as const;

const rnnBpttFigures = {
  "00 Учебник/Assets/Figures/curated/d2l-recurrent/unfolded-rnn.svg": {
    "name": "unfolded-rnn",
    "hash": "6f21d86aca0cba957ccd93e6e5904beacde382fbb97849ceeb6f79cc30d0437f",
    "page": "rnn",
    "viewBox": "0 0 390 148"
  },
  "00 Учебник/Assets/Figures/curated/d2l-recurrent/rnn-bptt.svg": {
    "name": "rnn-bptt",
    "hash": "edfd5bc723d46d6e5d6b6c18b7faf2652f284a7e24d78e20be7e5643315f2ae4",
    "page": "bptt",
    "viewBox": "0 0 303 132"
  },
  "00 Учебник/Assets/Figures/curated/rnn-bptt-2026-10-21/d2l-rnn.svg": {
    "name": "rnn",
    "hash": "d19bd5d2bf26ec93f87cbcdc72ef1373971f184345a5b856f4d6896269405545",
    "page": "rnn",
    "viewBox": "0 0 405 175"
  },
  "00 Учебник/Assets/Figures/curated/rnn-bptt-2026-10-21/d2l-rnn-train.svg": {
    "name": "rnn-train",
    "hash": "cf26c7c7dbfad4f8cb2a1f0876aa216327b006b52dda249c65a09f8d636bac46",
    "page": "rnn",
    "viewBox": "0 0 358 150"
  },
  "00 Учебник/Assets/Figures/curated/rnn-bptt-2026-10-21/d2l-truncated-bptt.svg": {
    "name": "truncated-bptt",
    "hash": "1fe6798ddee72f020f49d97d1b8fa65bd684010f2217156d6da9ad11487e7ba9",
    "page": "bptt",
    "viewBox": "0 0 255 82"
  }
} as const;

const lstmGruFigures = {
  "00 Учебник/Assets/Figures/curated/lstm-gru-author-2026-11-10/d2l-lstm-1.svg": {
    "id": "curated-d2l-lstm-1-82974021e95b",
    "path": "lstm-gru-author-2026-11-10/d2l-lstm-1.svg",
    "member": "d2l-lstm-1.svg",
    "sourceName": "lstm-1",
    "hash": "82974021e95bc18104f6fafff70c1621914fbea87cae579a2a84088bd7a3e720",
    "kind": "d2l",
    "page": "lstm",
    "viewBox": "0 0 327 208"
  },
  "00 Учебник/Assets/Figures/curated/lstm-gru-author-2026-11-10/d2l-lstm-2.svg": {
    "id": "curated-d2l-lstm-2-1e42c97a7ca5",
    "path": "lstm-gru-author-2026-11-10/d2l-lstm-2.svg",
    "member": "d2l-lstm-2.svg",
    "sourceName": "lstm-2",
    "hash": "1e42c97a7ca5c296db3888f93de979d45ad5ee5cf79f3b1d09d72ab5a5492883",
    "kind": "d2l",
    "page": "lstm",
    "viewBox": "0 0 414 214"
  },
  "00 Учебник/Assets/Figures/curated/lstm-gru-author-2026-11-10/d2l-gru-2.svg": {
    "id": "curated-d2l-gru-2-1cdd63f71aa7",
    "path": "lstm-gru-author-2026-11-10/d2l-gru-2.svg",
    "member": "d2l-gru-2.svg",
    "sourceName": "gru-2",
    "hash": "1cdd63f71aa7e75993d7fd314b4ad30d62e680ef13444330d21220d73d635eae",
    "kind": "d2l",
    "page": "gru",
    "viewBox": "0 0 414 212"
  },
  "00 Учебник/Assets/Figures/curated/d2l-recurrent/lstm-gates.svg": {
    "id": "curated-d2l-lstm-gates-00000000000c",
    "path": "d2l-recurrent/lstm-gates.svg",
    "member": "d2l-lstm-0.svg",
    "sourceName": "lstm-0",
    "hash": "4485cb112685a8fd06379de6ba092b67fab283b289397430cc72f4be371f7d1f",
    "kind": "d2l",
    "page": "lstm",
    "viewBox": "0 0 331 208",
    "oldMember": "lstm-0"
  },
  "00 Учебник/Assets/Figures/curated/d2l-recurrent/lstm-hidden-state.svg": {
    "id": "curated-d2l-lstm-state-00000000000d",
    "path": "d2l-recurrent/lstm-hidden-state.svg",
    "member": "d2l-lstm-3.svg",
    "sourceName": "lstm-3",
    "hash": "41fb0bc833e05ff2b24c86bf1b15b1c275009e7e2b4f61b7404df61290d95874",
    "kind": "d2l",
    "page": "lstm",
    "viewBox": "0 0 414 214",
    "oldMember": "lstm-3"
  },
  "00 Учебник/Assets/Figures/curated/d2l-recurrent/gru.svg": {
    "id": "curated-d2l-gru-00000000000e",
    "path": "d2l-recurrent/gru.svg",
    "member": "d2l-gru-1.svg",
    "sourceName": "gru-1",
    "hash": "8607d177a01254ba66989d4eb8e8801eb2b5962ce486b7c72d213efd660d02f7",
    "kind": "d2l",
    "page": "gru",
    "viewBox": "0 0 309 212",
    "oldMember": "gru-1"
  },
  "00 Учебник/Assets/Figures/curated/lstm-gru-2026-10-21/d2l-gru-3.svg": {
    "id": "curated-lstm-gru-d2l-full-278a159981ea",
    "path": "lstm-gru-2026-10-21/d2l-gru-3.svg",
    "member": "d2l-gru-3.svg",
    "sourceName": "gru-3",
    "hash": "278a159981eaed707a03cd85cf32380f1d2167c8b152b910b14ab052a0cf8ef9",
    "kind": "d2l",
    "page": "gru",
    "viewBox": "0 0 414 208"
  },
  "00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-chain.png": {
    "id": "curated-colah-lstm-chain-0000000000b8",
    "path": "source-audit-15-24/LSTM3-chain.png",
    "member": "LSTM3-chain.png",
    "sourceName": "LSTM3-chain",
    "hash": "9ef0aeb8ba4e3c55e5def82755276bcdd345f716dbb52b8eb437d8bbfd4b2e63",
    "kind": "colah",
    "width": 2233,
    "height": 839
  },
  "00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-focus-f.png": {
    "id": "curated-colah-lstm-forget-0000000000b9",
    "path": "source-audit-15-24/LSTM3-focus-f.png",
    "member": "LSTM3-focus-f.png",
    "sourceName": "LSTM3-focus-f",
    "hash": "8462a501feb0ff729732b2e4c50d8a0670b237f243cf456790df7c782beae828",
    "kind": "colah",
    "width": 1826,
    "height": 564
  },
  "00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-focus-i.png": {
    "id": "curated-colah-lstm-input-0000000000ba",
    "path": "source-audit-15-24/LSTM3-focus-i.png",
    "member": "LSTM3-focus-i.png",
    "sourceName": "LSTM3-focus-i",
    "hash": "8bc3dc738aeed13979e07240ccbf1d35bd1bf844e2f01a31bc67c4ee689aed61",
    "kind": "colah",
    "width": 1826,
    "height": 564
  },
  "00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-focus-C.png": {
    "id": "curated-colah-lstm-cell-update-0000000000bb",
    "path": "source-audit-15-24/LSTM3-focus-C.png",
    "member": "LSTM3-focus-C.png",
    "sourceName": "LSTM3-focus-C",
    "hash": "635f665c327bbe2434bc5aa9a8b078f29343e9c93a9b87c5930a59ae719aed39",
    "kind": "colah",
    "width": 1826,
    "height": 564
  },
  "00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-focus-o.png": {
    "id": "curated-colah-lstm-output-0000000000bc",
    "path": "source-audit-15-24/LSTM3-focus-o.png",
    "member": "LSTM3-focus-o.png",
    "sourceName": "LSTM3-focus-o",
    "hash": "d4929243878c8802b7d051ed078761293a7ef8dad3b994b170083ef1dbe42083",
    "kind": "colah",
    "width": 1826,
    "height": 564
  },
  "00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-C-line.png": {
    "id": "curated-colah-lstm-cell-line-0000000000bd",
    "path": "source-audit-15-24/LSTM3-C-line.png",
    "member": "LSTM3-C-line.png",
    "sourceName": "LSTM3-C-line",
    "hash": "141df91d2a36401c7f9f76b84ddee4ee9ebae19818ce4f0961f10a48295f205b",
    "kind": "colah",
    "width": 1826,
    "height": 564
  },
  "00 Учебник/Assets/Figures/curated/source-audit-15-24/LSTM3-var-GRU.png": {
    "id": "curated-colah-gru-0000000000be",
    "path": "source-audit-15-24/LSTM3-var-GRU.png",
    "member": "LSTM3-var-GRU.png",
    "sourceName": "LSTM3-var-GRU",
    "hash": "89ac067d7acec97b26f2c3083c19eeac56223778e3bd72893aaa37732b5e5f0f",
    "kind": "colah",
    "width": 1826,
    "height": 564
  }
} as const;

const seq2seqFigures = {
  "00 Учебник/Assets/Figures/curated/d2l-recurrent/seq2seq.svg": {
    "asset": "00 Учебник/Assets/Figures/curated/d2l-recurrent/seq2seq.svg",
    "id": "curated-d2l-seq2seq-00000000000f",
    "name": "seq2seq",
    "page": "seq2seq",
    "hash": "aebcb5867aaed90b6e71cc4a186248dded1668f4c617d58f51c2fb2ba479b775",
    "viewBox": "0 0 355 112",
    "url": "https://d2l.ai/chapter_recurrent-modern/seq2seq.html",
    "pageHash": "4a8968fc425166377153fd749b359411455918496498ab3c98374a7b367b6c04"
  },
  "00 Учебник/Assets/Figures/curated/d2l-recurrent/seq2seq-attention.svg": {
    "asset": "00 Учебник/Assets/Figures/curated/d2l-recurrent/seq2seq-attention.svg",
    "id": "curated-d2l-seq2seq-attn-000000000010",
    "name": "seq2seq-details-attention",
    "page": "bahdanau-attention",
    "hash": "5ca0c3979daa6de12925a6a00bdd33709218eb2c3501bf9247c89c63579066b4",
    "viewBox": "0 0 308 174",
    "url": "https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html",
    "pageHash": "8e4a4c470f0426475f4f0fc45821c52a0e244dd6171f439504084d2aa6013ab9"
  },
  "00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/seq2seq-details.svg": {
    "asset": "00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/seq2seq-details.svg",
    "id": "curated-seq2seq-seq2seq-details-6f49a4dcad55",
    "name": "seq2seq-details",
    "page": "seq2seq",
    "hash": "6f49a4dcad55ed4b86eaa826306fb58e16c487389aa295e3a0db20580354af81",
    "viewBox": "0 0 244 136",
    "url": "https://d2l.ai/chapter_recurrent-modern/seq2seq.html",
    "pageHash": "4a8968fc425166377153fd749b359411455918496498ab3c98374a7b367b6c04"
  },
  "00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/seq2seq-predict.svg": {
    "asset": "00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/seq2seq-predict.svg",
    "id": "curated-seq2seq-seq2seq-predict-e8de60b19aa2",
    "name": "seq2seq-predict",
    "page": "seq2seq",
    "hash": "e8de60b19aa2a90224bfedb59945c994963df45151bfa333e79d2abe6163f643",
    "viewBox": "0 0 355 112",
    "url": "https://d2l.ai/chapter_recurrent-modern/seq2seq.html",
    "pageHash": "4a8968fc425166377153fd749b359411455918496498ab3c98374a7b367b6c04"
  },
  "00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/s2s-prob1.svg": {
    "asset": "00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/s2s-prob1.svg",
    "id": "curated-seq2seq-s2s-prob1-125fe80f8e8c",
    "name": "s2s-prob1",
    "page": "beam-search",
    "hash": "125fe80f8e8c9c57625e485211f26d8df713ce336149690a01a501ce351c7ce7",
    "viewBox": "0 0 148 87",
    "url": "https://d2l.ai/chapter_recurrent-modern/beam-search.html",
    "pageHash": "b71d99c04c00dc5393991c6df4d36971e6d54e39853b01b200d4a2f038a3a833"
  },
  "00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/s2s-prob2.svg": {
    "asset": "00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/s2s-prob2.svg",
    "id": "curated-seq2seq-s2s-prob2-22108fb9b042",
    "name": "s2s-prob2",
    "page": "beam-search",
    "hash": "22108fb9b042ac81ca638962ff073a33bf9b58372ca9118c267a61eda95b1600",
    "viewBox": "0 0 148 87",
    "url": "https://d2l.ai/chapter_recurrent-modern/beam-search.html",
    "pageHash": "b71d99c04c00dc5393991c6df4d36971e6d54e39853b01b200d4a2f038a3a833"
  },
  "00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/beam-search.svg": {
    "asset": "00 Учебник/Assets/Figures/curated/seq2seq-2026-10-21/beam-search.svg",
    "id": "curated-seq2seq-beam-search-b3f9ed8af5a5",
    "name": "beam-search",
    "page": "beam-search",
    "hash": "b3f9ed8af5a51f7648c52b259b6317e546abde3be6275f594e28ee804488c731",
    "viewBox": "0 0 364 234",
    "url": "https://d2l.ai/chapter_recurrent-modern/beam-search.html",
    "pageHash": "b71d99c04c00dc5393991c6df4d36971e6d54e39853b01b200d4a2f038a3a833"
  }
} as const;

const attentionTranslationFigures = {
  "00 Учебник/Assets/Figures/curated/attention-is-all-you-need/bahdanau_examples-min.png": {
    "name": "bahdanau_examples-min.png",
    "kind": "voita",
    "id": "curated-attention-is-all-you-need-6c66971d0df0",
    "asset": "00 Учебник/Assets/Figures/curated/attention-is-all-you-need/bahdanau_examples-min.png",
    "sourceMember": "voita-bahdanau_examples-min.png",
    "hash": "6c66971d0df00e5535298faddda0c1bb00fd141d905a4f820d030674682728d5",
    "size": [
      1625,
      803
    ],
    "author": "Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (figure); Lena Voita (course reproduction)",
    "source": "05 Источники/Originals/attention-translation-2026-10-21/voita-bahdanau_examples-min.png",
    "sourceHash": "6c66971d0df00e5535298faddda0c1bb00fd141d905a4f820d030674682728d5",
    "url": "https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html",
    "image": "https://lena-voita.github.io/resources/lectures/seq2seq/attention/bahdanau_examples-min.png",
    "page": "05 Источники/Originals/seq2seq-2026-10-21/voita-seq2seq.html",
    "pageHash": "b9c976f69fc2f78a046352a58c69b1f5c7c38cd52b6b37c493cbbad885b6ff25",
    "license": "User permission record (not a named license)",
    "licenseURL": null,
    "status": "needs-confirmation",
    "rights": "source-verified; third-party-rights-not-independently-verified",
    "derivation": "source-raster-copy"
  },
  "00 Учебник/Assets/Figures/curated/attention-is-all-you-need/bottleneck-min.png": {
    "name": "bottleneck-min.png",
    "kind": "voita",
    "id": "curated-attention-is-all-you-need-4e357db2d637",
    "asset": "00 Учебник/Assets/Figures/curated/attention-is-all-you-need/bottleneck-min.png",
    "sourceMember": "voita-bottleneck-min.png",
    "hash": "4e357db2d637f8bc0782974e00cf1b35bb56b5bcac67b35e84115b8c9ef64e82",
    "size": [
      1268,
      513
    ],
    "author": "Lena Voita",
    "source": "05 Источники/Originals/attention-translation-2026-10-21/voita-bottleneck-min.png",
    "sourceHash": "4e357db2d637f8bc0782974e00cf1b35bb56b5bcac67b35e84115b8c9ef64e82",
    "url": "https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html",
    "image": "https://lena-voita.github.io/resources/lectures/seq2seq/attention/bottleneck-min.png",
    "page": "05 Источники/Originals/seq2seq-2026-10-21/voita-seq2seq.html",
    "pageHash": "b9c976f69fc2f78a046352a58c69b1f5c7c38cd52b6b37c493cbbad885b6ff25",
    "license": "User permission record (not a named license)",
    "licenseURL": null,
    "status": "needs-confirmation",
    "rights": "source-verified; third-party-rights-not-independently-verified",
    "derivation": "source-raster-copy"
  },
  "00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/luong-input-feeding.png": {
    "name": "luong-input-feeding.png",
    "kind": "acl",
    "id": "curated-attention-translation-luong-input-feeding-png-23119692a647",
    "asset": "00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/luong-input-feeding.png",
    "hash": "23119692a647c586ddf1955350b707e39ea0bc785e4dc239b5283639103f9520",
    "size": [
      915,
      938
    ],
    "source": "05 Источники/Originals/attention-translation-2026-10-21/luong-D15-1166.pdf",
    "sourceHash": "e8aca09523b8807edcafb48b8ee9fa60478e9be3a3682f9ffa61fd00de40e79a",
    "url": "https://aclanthology.org/D15-1166/",
    "image": "https://aclanthology.org/D15-1166.pdf",
    "author": "Minh-Thang Luong, Hieu Pham, Christopher D. Manning",
    "page": "05 Источники/Originals/attention-translation-2026-10-21/acl-D15-1166.html",
    "pageHash": "44f303cd79a25692e0afd42ebfe7e993d55f6fe6196bc7edd565ecfbe7c39ed8",
    "license": "CC BY-NC-SA 3.0",
    "licenseURL": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
    "evidence": "05 Источники/Originals/attention-translation-2026-10-21/acl-D15-1166.html",
    "evidenceHash": "44f303cd79a25692e0afd42ebfe7e993d55f6fe6196bc7edd565ecfbe7c39ed8",
    "status": "verified",
    "derivation": "pdf-page-render-crop",
    "crop": [
      405,
      270,
      1320,
      1208
    ],
    "pageNumber": 5,
    "printed": 1416
  },
  "00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/luong-global.png": {
    "name": "luong-global.png",
    "kind": "acl",
    "id": "curated-attention-translation-luong-global-png-324515ce7fbc",
    "asset": "00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/luong-global.png",
    "hash": "324515ce7fbc1498e88de8218bb7d15d4de56f068ad8a0ffddbab22d9b3b453d",
    "size": [
      1118,
      968
    ],
    "source": "05 Источники/Originals/attention-translation-2026-10-21/luong-D15-1166.pdf",
    "sourceHash": "e8aca09523b8807edcafb48b8ee9fa60478e9be3a3682f9ffa61fd00de40e79a",
    "url": "https://aclanthology.org/D15-1166/",
    "image": "https://aclanthology.org/D15-1166.pdf",
    "author": "Minh-Thang Luong, Hieu Pham, Christopher D. Manning",
    "page": "05 Источники/Originals/attention-translation-2026-10-21/acl-D15-1166.html",
    "pageHash": "44f303cd79a25692e0afd42ebfe7e993d55f6fe6196bc7edd565ecfbe7c39ed8",
    "license": "CC BY-NC-SA 3.0",
    "licenseURL": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
    "evidence": "05 Источники/Originals/attention-translation-2026-10-21/acl-D15-1166.html",
    "evidenceHash": "44f303cd79a25692e0afd42ebfe7e993d55f6fe6196bc7edd565ecfbe7c39ed8",
    "status": "verified",
    "derivation": "pdf-page-render-crop",
    "crop": [
      1522,
      240,
      2640,
      1208
    ],
    "pageNumber": 3,
    "printed": 1414
  },
  "00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/d2l-attention-output.svg": {
    "name": "d2l-attention-output.svg",
    "kind": "d2l",
    "id": "curated-attention-translation-d2l-attention-output-svg-be729a3c5680",
    "asset": "00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/d2l-attention-output.svg",
    "hash": "be729a3c5680d6ccfb46d0c57b1af39e3ed36ff399c3eaa1c26292ae5b733bdf",
    "size": null,
    "source": "05 Источники/Originals/attention-translation-2026-10-21/d2l-attention-output.svg",
    "sourceHash": "be729a3c5680d6ccfb46d0c57b1af39e3ed36ff399c3eaa1c26292ae5b733bdf",
    "url": "https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html",
    "image": "https://raw.githubusercontent.com/d2l-ai/d2l-en/23d7a5aecceee57d1292c56e90cce307f183bb0a/img/attention-output.svg",
    "commit": "23d7a5aecceee57d1292c56e90cce307f183bb0a",
    "author": "Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola",
    "page": "05 Источники/Originals/attention-translation-2026-10-21/d2l-attention-scoring-functions.md",
    "pageHash": "0ff7897c34919301660855ae6820173b657dd28926f7de4ae3ec527454c4285d",
    "license": "CC BY-SA 4.0",
    "licenseURL": "https://creativecommons.org/licenses/by-sa/4.0/",
    "evidence": "05 Источники/Originals/neural-language-model-2026-10-21/d2l-LICENSE.txt",
    "evidenceHash": "a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c",
    "status": "verified",
    "derivation": "source-svg-copy",
    "viewBox": "0 0 349 181"
  },
  "00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/distill-rnn_read.svg": {
    "name": "distill-rnn_read.svg",
    "kind": "distill",
    "id": "curated-attention-translation-distill-rnn-read-svg-36dc327998c7",
    "asset": "00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/distill-rnn_read.svg",
    "hash": "36dc327998c717ab14f234288cc8ddf6544669e9acc222c0c6be9e10f55159ad",
    "size": null,
    "source": "05 Источники/Originals/attention-translation-2026-10-21/distill-rnn_read.svg",
    "sourceHash": "36dc327998c717ab14f234288cc8ddf6544669e9acc222c0c6be9e10f55159ad",
    "url": "https://distill.pub/2016/augmented-rnns/",
    "image": "https://raw.githubusercontent.com/distillpub/post--augmented-rnns/fdc3526167133f5a97cee9e7cb48af293e67d66d/public/assets/rnn_read.svg",
    "commit": "fdc3526167133f5a97cee9e7cb48af293e67d66d",
    "author": "Chris Olah, Shan Carter",
    "page": "05 Источники/Originals/attention-translation-2026-10-21/distill-index-pinned.html",
    "pageHash": "16e80facad66e6a9d46a4123b016ddb8ae4a4563558492b3f6dda4dba2f9d25b",
    "license": "CC BY 4.0",
    "licenseURL": "https://creativecommons.org/licenses/by/4.0/",
    "evidence": "05 Источники/Originals/attention-translation-2026-10-21/distill-README.md",
    "evidenceHash": "2f9b0e90c03611f38aef12ce8305fdf3dc3e6fa4c956ac673a9c738e4d0a2a1e",
    "status": "verified",
    "derivation": "source-svg-copy",
    "viewBox": "0 0 744 252"
  },
  "00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/luong-local.png": {
    "name": "luong-local.png",
    "kind": "acl",
    "id": "curated-attention-translation-luong-local-png-340def2c29c9",
    "asset": "00 Учебник/Assets/Figures/curated/attention-translation-2026-10-21/luong-local.png",
    "hash": "340def2c29c97dddbd44723356dcb4c07471249ef6c5d8a280d40f6fc407acd8",
    "size": [
      1125,
      968
    ],
    "source": "05 Источники/Originals/attention-translation-2026-10-21/luong-D15-1166.pdf",
    "sourceHash": "e8aca09523b8807edcafb48b8ee9fa60478e9be3a3682f9ffa61fd00de40e79a",
    "url": "https://aclanthology.org/D15-1166/",
    "image": "https://aclanthology.org/D15-1166.pdf",
    "author": "Minh-Thang Luong, Hieu Pham, Christopher D. Manning",
    "page": "05 Источники/Originals/attention-translation-2026-10-21/acl-D15-1166.html",
    "pageHash": "44f303cd79a25692e0afd42ebfe7e993d55f6fe6196bc7edd565ecfbe7c39ed8",
    "license": "CC BY-NC-SA 3.0",
    "licenseURL": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
    "evidence": "05 Источники/Originals/attention-translation-2026-10-21/acl-D15-1166.html",
    "evidenceHash": "44f303cd79a25692e0afd42ebfe7e993d55f6fe6196bc7edd565ecfbe7c39ed8",
    "status": "verified",
    "derivation": "pdf-page-render-crop",
    "crop": [
      315,
      240,
      1440,
      1208
    ],
    "pageNumber": 4,
    "printed": 1415
  }
} as const;

const selfAttentionFigures = {
  "00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png": {
    "fields": {
      "id": "curated-self-attention-alammar-vectors-d6415b63f3cf",
      "author": "Jay Alammar",
      "source_url": "https://jalammar.github.io/illustrated-transformer/",
      "license": "CC BY-NC-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
      "source_page_snapshot": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "source_page_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "license_evidence": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "license_evidence_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "derivation": "source-png-copy",
      "provenance_confirmation": "self-attention-source-snapshot",
      "metadata_status": "verified",
      "source_image_url": "https://jalammar.github.io/images/t/transformer_self_attention_vectors.png",
      "source_asset": "05 Источники/Originals/self-attention-2026-10-21/transformer_self_attention_vectors.png",
      "source_asset_sha256": "d6415b63f3cfb5d802719aebf111283bbcdab3cb33b40194f3b2d8a65522117c",
      "sha256": "d6415b63f3cfb5d802719aebf111283bbcdab3cb33b40194f3b2d8a65522117c"
    },
    "size": [
      875,
      552
    ]
  },
  "00 Учебник/Assets/Figures/attention/self-attention-matrix-calculation-2.png": {
    "fields": {
      "id": "curated-self-attention-alammar-matrix-fd6e2158c1ec",
      "author": "Jay Alammar",
      "source_url": "https://jalammar.github.io/illustrated-transformer/",
      "license": "CC BY-NC-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
      "source_page_snapshot": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "source_page_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "license_evidence": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "license_evidence_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "derivation": "source-png-copy",
      "provenance_confirmation": "self-attention-source-snapshot",
      "metadata_status": "verified",
      "source_image_url": "https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png",
      "source_asset": "05 Источники/Originals/self-attention-2026-10-21/self-attention-matrix-calculation-2.png",
      "source_asset_sha256": "fd6e2158c1ec73294a2915b5637d77f6913fe361316b3df397c398cc6128b8a1",
      "sha256": "fd6e2158c1ec73294a2915b5637d77f6913fe361316b3df397c398cc6128b8a1"
    },
    "size": [
      893,
      349
    ]
  },
  "00 Учебник/Assets/Figures/curated/self-attention-2026-10-21/self-attention-output.png": {
    "fields": {
      "id": "curated-self-attention-alammar-output-0818b251e33e",
      "author": "Jay Alammar",
      "source_url": "https://jalammar.github.io/illustrated-transformer/",
      "license": "CC BY-NC-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
      "source_page_snapshot": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "source_page_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "license_evidence": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "license_evidence_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "derivation": "source-png-copy",
      "provenance_confirmation": "self-attention-source-snapshot",
      "metadata_status": "verified",
      "source_image_url": "https://jalammar.github.io/images/t/self-attention-output.png",
      "source_asset": "05 Источники/Originals/self-attention-2026-10-21/self-attention-output.png",
      "source_asset_sha256": "0818b251e33eee8d20ff69235d87bab54588e4abbcab2d7a12eb2d05d3f83709",
      "sha256": "0818b251e33eee8d20ff69235d87bab54588e4abbcab2d7a12eb2d05d3f83709"
    },
    "size": [
      786,
      747
    ]
  },
  "00 Учебник/Assets/Figures/curated/self-attention-2026-10-21/cnn-rnn-self-attention.svg": {
    "fields": {
      "id": "curated-self-attention-d2l-comparison-937aba60f212",
      "author": "Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola",
      "source_url": "https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html",
      "source_image_url": "https://raw.githubusercontent.com/d2l-ai/d2l-en/23d7a5aecceee57d1292c56e90cce307f183bb0a/img/cnn-rnn-self-attention.svg",
      "commit": "23d7a5aecceee57d1292c56e90cce307f183bb0a",
      "license": "CC BY-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
      "source_asset": "05 Источники/Originals/self-attention-2026-10-21/cnn-rnn-self-attention.svg",
      "source_asset_sha256": "937aba60f21259a298a58f30e581a3e6d9989901f92a827aa9efbcdfd9e989ae",
      "sha256": "937aba60f21259a298a58f30e581a3e6d9989901f92a827aa9efbcdfd9e989ae",
      "source_page_snapshot": "05 Источники/Originals/self-attention-2026-10-21/d2l-self-attention.md",
      "source_page_sha256": "8b6eeedba841fbd231edaa5f0f1524bf17c82449623bd2e46f8da21a0cead35f",
      "license_evidence": "05 Источники/Originals/neural-language-model-2026-10-21/d2l-LICENSE.txt",
      "license_evidence_sha256": "a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c",
      "derivation": "source-svg-copy",
      "provenance_confirmation": "self-attention-source-snapshot",
      "metadata_status": "verified"
    },
    "viewBox": "0 0 377 251"
  }
} as const;

const maskingHeadsFigures = {
  "00 Учебник/Assets/Figures/curated/source-audit-15-24/causal-mask-detailed.png": {
    "fields": {
      "id": "curated-causal-mask-godoy-0000000000c1",
      "author": "dvgodoy",
      "source_url": "https://commons.wikimedia.org/wiki/File:Decoder_self-attention_with_causal_masking,_detailed_diagram.png",
      "source_image_url": "https://upload.wikimedia.org/wikipedia/commons/7/72/Decoder_self-attention_with_causal_masking%2C_detailed_diagram.png",
      "license": "CC BY 4.0",
      "license_url": "https://creativecommons.org/licenses/by/4.0/",
      "source_asset": "05 Источники/Originals/masking-heads-2026-10-21/causal-mask-detailed.png",
      "source_asset_sha256": "8774ac0e35806218e665d433fcf82a5c0dcf69ade2c52ed6f656e2c57d42600d",
      "sha256": "8774ac0e35806218e665d433fcf82a5c0dcf69ade2c52ed6f656e2c57d42600d",
      "source_page_snapshot": "05 Источники/Originals/masking-heads-2026-10-21/commons-causal-mask.html",
      "source_page_sha256": "aff9b6d2532d0425fd79c16ba631e30d8b53243a6c338b1de1d4d911e1b13935",
      "license_evidence": "05 Источники/Originals/masking-heads-2026-10-21/commons-causal-mask.html",
      "license_evidence_sha256": "aff9b6d2532d0425fd79c16ba631e30d8b53243a6c338b1de1d4d911e1b13935",
      "derivation": "source-png-copy",
      "provenance_confirmation": "masking-heads-source-snapshot",
      "metadata_status": "verified"
    },
    "size": [
      1426,
      1000
    ]
  },
  "00 Учебник/Assets/Figures/curated/source-audit-15-24/qkv-for-heads.png": {
    "fields": {
      "id": "curated-voita-qkv-heads-0000000000c2",
      "author": "Elena Voita",
      "source_url": "https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#multi_head_attention",
      "source_image_url": "https://lena-voita.github.io/resources/lectures/seq2seq/transformer/qkv_for_heads-min.png",
      "license": "User permission record (not a named license)",
      "license_url": null,
      "source_asset": "05 Источники/Originals/masking-heads-2026-10-21/voita-qkv_for_heads-min.png",
      "source_asset_sha256": "aa8c7d8d37b41a40eb4463ad904dcead83470c901fc2bfdc7eda64d2f38776f8",
      "sha256": "aa8c7d8d37b41a40eb4463ad904dcead83470c901fc2bfdc7eda64d2f38776f8",
      "source_page_snapshot": "05 Источники/Originals/seq2seq-2026-10-21/voita-seq2seq.html",
      "source_page_sha256": "b9c976f69fc2f78a046352a58c69b1f5c7c38cd52b6b37c493cbbad885b6ff25",
      "rights_status": "source-verified; third-party-rights-not-independently-verified",
      "derivation": "source-png-copy",
      "provenance_confirmation": "masking-heads-source-snapshot",
      "metadata_status": "needs-confirmation"
    },
    "size": [
      911,
      689
    ]
  },
  "00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_qkv.png": {
    "fields": {
      "id": "curated-alammar-heads-qkv-0000000000c3",
      "author": "Jay Alammar",
      "source_url": "https://jalammar.github.io/illustrated-transformer/",
      "source_image_url": "https://jalammar.github.io/images/t/transformer_attention_heads_qkv.png",
      "license": "CC BY-NC-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
      "source_asset": "05 Источники/Originals/masking-heads-2026-10-21/transformer_attention_heads_qkv.png",
      "source_asset_sha256": "d360b9199193a6db44e8608c80f0987a5bc824a2d1061d425d22b49f71981aca",
      "sha256": "d360b9199193a6db44e8608c80f0987a5bc824a2d1061d425d22b49f71981aca",
      "source_page_snapshot": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "source_page_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "license_evidence": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "license_evidence_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "derivation": "source-png-copy",
      "provenance_confirmation": "masking-heads-source-snapshot",
      "metadata_status": "verified"
    },
    "size": [
      1310,
      774
    ]
  },
  "00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_z.png": {
    "fields": {
      "id": "curated-alammar-heads-z-0000000000c4",
      "author": "Jay Alammar",
      "source_url": "https://jalammar.github.io/illustrated-transformer/",
      "source_image_url": "https://jalammar.github.io/images/t/transformer_attention_heads_z.png",
      "license": "CC BY-NC-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
      "source_asset": "05 Источники/Originals/masking-heads-2026-10-21/transformer_attention_heads_z.png",
      "source_asset_sha256": "d198af28a747da599f9f79c202a98cebbe4b69304cdcc7ef2fe33bca18e7f36c",
      "sha256": "d198af28a747da599f9f79c202a98cebbe4b69304cdcc7ef2fe33bca18e7f36c",
      "source_page_snapshot": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "source_page_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "license_evidence": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "license_evidence_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "derivation": "source-png-copy",
      "provenance_confirmation": "masking-heads-source-snapshot",
      "metadata_status": "verified"
    },
    "size": [
      1018,
      483
    ]
  },
  "00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_weight_matrix_o.png": {
    "fields": {
      "id": "curated-alammar-heads-wo-0000000000c5",
      "author": "Jay Alammar",
      "source_url": "https://jalammar.github.io/illustrated-transformer/",
      "source_image_url": "https://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png",
      "license": "CC BY-NC-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
      "source_asset": "05 Источники/Originals/masking-heads-2026-10-21/transformer_attention_heads_weight_matrix_o.png",
      "source_asset_sha256": "606c0e2a4d2c4dba858496226a1040dcd374122d2a2a3568390a06915edfaba7",
      "sha256": "606c0e2a4d2c4dba858496226a1040dcd374122d2a2a3568390a06915edfaba7",
      "source_page_snapshot": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "source_page_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "license_evidence": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "license_evidence_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "derivation": "source-png-copy",
      "provenance_confirmation": "masking-heads-source-snapshot",
      "metadata_status": "verified"
    },
    "size": [
      1372,
      759
    ]
  }
} as const;

const positionalFigures = {
  "00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_positional_encoding_vectors.png": {
    "size": [
      1438,
      793
    ],
    "fields": {
      "id": "curated-attention-is-all-you-need-1447f2a6561e",
      "author": "Jay Alammar",
      "source_url": "https://jalammar.github.io/illustrated-transformer/",
      "source_image_url": "https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png",
      "license": "CC BY-NC-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
      "source_asset": "05 Источники/Originals/positional-information-2026-10-21/alammar-positional-vectors.png",
      "source_asset_sha256": "1447f2a6561e5aef3ca9ab3792490d744b81c34c873d19b1396ed770ca19b669",
      "sha256": "1447f2a6561e5aef3ca9ab3792490d744b81c34c873d19b1396ed770ca19b669",
      "source_page_snapshot": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "source_page_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "license_evidence": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "license_evidence_sha256": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556",
      "derivation": "source-png-copy",
      "metadata_status": "verified",
      "provenance_confirmation": "positional-information-source-snapshot"
    }
  },
  "00 Учебник/Assets/Figures/curated/source-audit-25-31/d2l-sinusoidal-curves.svg": {
    "viewBox": "0 0 507.310425 244.474996",
    "fields": {
      "id": "curated-d2l-sinusoidal-curves-0000000000c7",
      "author": "Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola",
      "source_url": "https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html",
      "source_image_url": "https://d2l.ai/_images/output_self-attention-and-positional-encoding_ce9eb6_51_0.svg",
      "license": "CC BY-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
      "source_asset": "05 Источники/Originals/positional-information-2026-10-21/d2l-output-51.svg",
      "source_asset_sha256": "f4d1aeebd58a4444284fd4d8e87a95e8dcb53efe3ad3d7d7bb41cd5fbb08b40a",
      "sha256": "f4d1aeebd58a4444284fd4d8e87a95e8dcb53efe3ad3d7d7bb41cd5fbb08b40a",
      "source_page_snapshot": "05 Источники/Originals/positional-information-2026-10-21/d2l-positional.html",
      "source_page_sha256": "69d56f3e9b8e883ae27c4875c6ccdb8acff6b4de0a1cd313d65a8e487512c1c5",
      "license_evidence": "05 Источники/Originals/neural-language-model-2026-10-21/d2l-LICENSE.txt",
      "license_evidence_sha256": "a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c",
      "derivation": "source-svg-copy",
      "metadata_status": "verified",
      "provenance_confirmation": "positional-information-source-snapshot"
    }
  },
  "00 Учебник/Assets/Figures/curated/source-audit-25-31/d2l-positional-heatmap.svg": {
    "viewBox": "0 0 285.034322 357.956624",
    "fields": {
      "id": "curated-d2l-positional-heatmap-0000000000c8",
      "author": "Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola",
      "source_url": "https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html",
      "source_image_url": "https://d2l.ai/_images/output_self-attention-and-positional-encoding_ce9eb6_81_0.svg",
      "license": "CC BY-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
      "source_asset": "05 Источники/Originals/positional-information-2026-10-21/d2l-output-81.svg",
      "source_asset_sha256": "c10d8e03aab02bbe6f24f9851e64674231c2211f34df154705e3f882031ea26d",
      "sha256": "c10d8e03aab02bbe6f24f9851e64674231c2211f34df154705e3f882031ea26d",
      "source_page_snapshot": "05 Источники/Originals/positional-information-2026-10-21/d2l-positional.html",
      "source_page_sha256": "69d56f3e9b8e883ae27c4875c6ccdb8acff6b4de0a1cd313d65a8e487512c1c5",
      "license_evidence": "05 Источники/Originals/neural-language-model-2026-10-21/d2l-LICENSE.txt",
      "license_evidence_sha256": "a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c",
      "derivation": "source-svg-copy",
      "metadata_status": "verified",
      "provenance_confirmation": "positional-information-source-snapshot"
    }
  },
  "00 Учебник/Assets/Figures/curated/positions-2026-10-21/shaw-relative-figure1.png": {
    "size": [
      1130,
      840
    ],
    "fields": {
      "id": "curated-positional-shaw-5c73df1712d9",
      "author": "Peter Shaw, Jakob Uszkoreit, Ashish Vaswani",
      "source_url": "https://aclanthology.org/N18-2074.pdf#page=3",
      "license": "CC BY 4.0",
      "license_url": "https://creativecommons.org/licenses/by/4.0/",
      "source_asset": "05 Источники/Originals/positional-information-2026-10-21/shaw-N18-2074.pdf",
      "source_asset_sha256": "f10b5d840c34d69e3d145dc20de8f668535dca23a95e0025200797956f268587",
      "sha256": "5c73df1712d9b4c44e017c8de539ba0726720c3692c2c70047dd43e69e3dd16c",
      "source_page_snapshot": "05 Источники/Originals/positional-information-2026-10-21/shaw-abstract.html",
      "source_page_sha256": "ce065daf20219dfb24a67bb1d6c297b9cb10f7085ead7e28b0059c4f017dec8f",
      "license_evidence": "05 Источники/Originals/positional-information-2026-10-21/acl-copyright.html",
      "license_evidence_sha256": "88d644d2e21542ef1cc6c304c264ce7543d81e87c9599aff7e1c68e7a6df9558",
      "source_page": 3,
      "source_printed_page": 466,
      "render_dpi": 360,
      "render_page_size_px": [
        2977,
        4210
      ],
      "crop_box_px": [
        345,
        300,
        1475,
        1140
      ],
      "derivation": "pdf-page-render-crop",
      "metadata_status": "verified",
      "provenance_confirmation": "positional-information-source-snapshot"
    }
  },
  "00 Учебник/Assets/Figures/curated/positions-2026-10-21/roformer-figure1.png": {
    "size": [
      1670,
      1065
    ],
    "fields": {
      "id": "curated-positional-roformer-b1a65eb5e30f",
      "author": "Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu",
      "source_url": "https://arxiv.org/pdf/2104.09864v5#page=5",
      "license": "CC BY-NC-ND 4.0",
      "license_url": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
      "source_asset": "05 Источники/Originals/positional-information-2026-10-21/roformer-v5.pdf",
      "source_asset_sha256": "e9a481fbe1c8a20b7b1fa566b13102a1896c7829fa9a8b4c80528452a5ddaf79",
      "sha256": "b1a65eb5e30f1510dce571cedb548b98c1db5c5e57ca04112b856f931a0d2591",
      "source_page_snapshot": "05 Источники/Originals/positional-information-2026-10-21/roformer-v5-abstract.html",
      "source_page_sha256": "aa47247bbd7f9b18970e8f6adc201cb641ff7bd461ba7e2814042f15259d9043",
      "license_evidence": "05 Источники/Originals/positional-information-2026-10-21/roformer-v5-abstract.html",
      "license_evidence_sha256": "aa47247bbd7f9b18970e8f6adc201cb641ff7bd461ba7e2814042f15259d9043",
      "source_page": 5,
      "source_printed_page": 5,
      "render_dpi": 360,
      "render_page_size_px": [
        3060,
        3960
      ],
      "crop_box_px": [
        705,
        2010,
        2375,
        3075
      ],
      "derivation": "pdf-page-render-crop",
      "metadata_status": "verified",
      "provenance_confirmation": "positional-information-source-snapshot"
    }
  },
  "00 Учебник/Assets/Figures/curated/positions-2026-10-21/alibi-original.jpeg": {
    "size": [
      872,
      354
    ],
    "fields": {
      "id": "curated-positional-alibi-ade378177ac2",
      "author": "Ofir Press, Noah A. Smith, Mike Lewis",
      "source_url": "https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/README.md",
      "source_image_url": "https://raw.githubusercontent.com/ofirpress/attention_with_linear_biases/4b92f28a005ead2567abe2359f633e73e08f3833/.github/ALiBi.jpeg",
      "commit": "4b92f28a005ead2567abe2359f633e73e08f3833",
      "license": "MIT",
      "license_url": "https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/LICENSE",
      "source_asset": "05 Источники/Originals/positional-information-2026-10-21/alibi-original.jpeg",
      "source_asset_sha256": "ade378177ac2b32c4a7319c3bee16ee61da4ec9aaf7d1f109bd0933d900b6cc7",
      "sha256": "ade378177ac2b32c4a7319c3bee16ee61da4ec9aaf7d1f109bd0933d900b6cc7",
      "source_page_snapshot": "05 Источники/Originals/positional-information-2026-10-21/alibi-README.md",
      "source_page_sha256": "982219168b454a84d43547c77f2616f65c39513045c3e8b8530e7af44cc14bd8",
      "license_evidence": "05 Источники/Originals/positional-information-2026-10-21/alibi-MIT.txt",
      "license_evidence_sha256": "52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85",
      "license_copy": "05 Источники/Originals/positional-information-2026-10-21/alibi-MIT.txt",
      "public_license_copy": "site/public/licenses/alibi-4b92f28a-MIT.txt",
      "derivation": "source-jpeg-copy",
      "metadata_status": "verified",
      "provenance_confirmation": "positional-information-source-snapshot"
    }
  }
} as const;

const originalScalingMetricFigures = [
    {"id":"curated-scaling-emergence-wei-d905d44f12c6","asset":"00 Учебник/Assets/Figures/curated/scaling-emergence-2026-11-09/wei-emergence-figure2.png","author":"Jason Wei, Yi Tay, Rishi Bommasani et al.","source_url":"https://arxiv.org/html/2206.07682v2#S3.F2","source_image_url":"https://raw.githubusercontent.com/stanford-cs336/lectures/8b59b50730766695c2ffedd1a79c50cd09b9eb91/images/wei-emergence-plot.png","source_asset":"05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/images/wei-emergence-plot.png","commit":"8b59b50730766695c2ffedd1a79c50cd09b9eb91","sha256":"d905d44f12c6861f6e1febd0af3e7ab446f1ccf1531a1702c43dd4b977dd453b","dimensions_px":[777,549],"license":"CC BY 4.0","license_url":"https://creativecommons.org/licenses/by/4.0/","rights_evidence":"https://arxiv.org/html/2206.07682v2","derivation":"source-raster-copy"},
    {"id":"curated-scaling-emergence-schaeffer-19c9799140b9","asset":"00 Учебник/Assets/Figures/curated/scaling-emergence-2026-11-09/schaeffer-multiplication-exact-match.svg","author":"Rylan Schaeffer, Brando Miranda and Sanmi Koyejo","source_url":"https://arxiv.org/html/2304.15004v2#S3.F3","source_image_url":"https://arxiv.org/html/2304.15004v2/multiplication_acc_vs_model_size_by_temp_by_target_str_len.svg","source_asset":"00 Учебник/Assets/Figures/curated/scaling-emergence-2026-11-09/schaeffer-multiplication-exact-match.svg","sha256":"19c9799140b911f3d7d6eb8a8f0194f154b12800ca89012c6336aac166d77054","license":"CC BY-NC-ND 4.0","license_url":"https://creativecommons.org/licenses/by-nc-nd/4.0/","rights_evidence":"https://arxiv.org/html/2304.15004v2","derivation":"source-svg-copy"},
    {"id":"curated-scaling-emergence-schaeffer-19ed0bcfe5a8","asset":"00 Учебник/Assets/Figures/curated/scaling-emergence-2026-11-09/schaeffer-multiplication-token-edits.svg","author":"Rylan Schaeffer, Brando Miranda and Sanmi Koyejo","source_url":"https://arxiv.org/html/2304.15004v2#S3.F3","source_image_url":"https://arxiv.org/html/2304.15004v2/multiplication_neg_token_edit_dist_vs_model_size_by_temp_by_target_str_len.svg","source_asset":"00 Учебник/Assets/Figures/curated/scaling-emergence-2026-11-09/schaeffer-multiplication-token-edits.svg","sha256":"19ed0bcfe5a86739ae0c656e516fdc28c242156c5d23b60ca2aed548c40f9aab","license":"CC BY-NC-ND 4.0","license_url":"https://creativecommons.org/licenses/by-nc-nd/4.0/","rights_evidence":"https://arxiv.org/html/2304.15004v2","derivation":"source-svg-copy"},
];

const isoflopOriginalDir = '00 Учебник/Assets/Figures/curated/scaling-isoflop-2026-11-09';
const isoflopOriginals = [
  ["happy_path.svg","0ac01e9299c3c92fc3705dc0a2482e1979e9d721dcff70935adf2efdb45df8ad","https://arxiv.org/html/2603.22339v1#S4.F3","https://arxiv.org/html/2603.22339v1/happy_path.svg"],
  ["asymmetric.svg","6df85f0b6ddf5d8be2486e2ea614d6aa37d60076e0f30072ab53b14c34fa8ebc","https://arxiv.org/html/2603.22339v1#S5.F4","https://arxiv.org/html/2603.22339v1/asymmetric.svg"],
  ["training_flop_vs_model_size.png","3fbaefa50fa091f9dc58136ae9772a1d27123bf9399751bc8be0c919a2be7841","https://arxiv.org/html/2404.10102v2#S2.F2","https://arxiv.org/html/2404.10102v2/training_flop_vs_model_size.png"],
  ["distributions.svg","98769023cbf1cd7193d454cab36051328d71182d12bb0a9895c2e4d8df6bd1e5","https://arxiv.org/html/2404.10102v2#S3.F3","https://arxiv.org/html/2404.10102v2/distributions.svg"],
  ["tokens_to_params_ratio_plot.svg","c550efc1565c868ac3ebb095a2f46a14f8fdd0ceb16246dcbab7d67028461cd6","https://arxiv.org/html/2404.10102v2#S3.F5","https://arxiv.org/html/2404.10102v2/tokens_to_params_ratio_plot.svg"],
] as const;

const seq2seqAttentionBridge = [
  {"asset":"00 Учебник/Assets/Figures/curated/seq2seq-attention-bridge-2026-11-10/d2l-seq2seq-state.svg","sha256":"02a4827cfcdbdde12e6c2c4af95357b0c66342f4fded2c020883e7c8d34e5ff6","source_url":"https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html#fig-s2s-attention-state","source_image_url":"https://raw.githubusercontent.com/d2l-ai/d2l-en/23d7a5aecceee57d1292c56e90cce307f183bb0a/img/seq2seq-state.svg","source_asset":"05 Источники/Originals/seq2seq-2026-10-21/seq2seq-state.svg","source_asset_sha256":"02a4827cfcdbdde12e6c2c4af95357b0c66342f4fded2c020883e7c8d34e5ff6","source_page_snapshot":"05 Источники/Originals/seq2seq-attention-bridge-2026-11-10/d2l-bahdanau.html","source_page_sha256":"ca248f2e2102f9dadc9ca2828327529c57e2f2f3837ef542a5470d021b0f8ae9","license":"CC BY-SA 4.0","license_url":"https://creativecommons.org/licenses/by-sa/4.0/","license_evidence":"05 Источники/Originals/neural-language-model-2026-10-21/d2l-LICENSE.txt","license_evidence_sha256":"a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c","rights_status":"source-license-verified","metadata_status":"verified","derivation":"source-svg-copy","viewBox":[0,0,304,136],"commit":"23d7a5aecceee57d1292c56e90cce307f183bb0a","used_in":["00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer.md"]},
  {"asset":"00 Учебник/Assets/Figures/curated/seq2seq-attention-bridge-2026-11-10/d2l-bahdanau-pytorch-weights.svg","sha256":"e4e45f9cff346aabba76206352c3d45afb87c1630a9189b116ebca7a79cf41b3","source_url":"https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html#training","source_image_url":"https://en.d2l.ai/_images/output_bahdanau-attention_dd4710_90_0.svg","source_asset":"05 Источники/Originals/seq2seq-attention-bridge-2026-11-10/d2l-bahdanau-pytorch-weights.svg","source_asset_sha256":"e4e45f9cff346aabba76206352c3d45afb87c1630a9189b116ebca7a79cf41b3","source_page_snapshot":"05 Источники/Originals/seq2seq-attention-bridge-2026-11-10/d2l-bahdanau.html","source_page_sha256":"ca248f2e2102f9dadc9ca2828327529c57e2f2f3837ef542a5470d021b0f8ae9","license":"CC BY-SA 4.0","license_url":"https://creativecommons.org/licenses/by-sa/4.0/","license_evidence":"05 Источники/Originals/neural-language-model-2026-10-21/d2l-LICENSE.txt","license_evidence_sha256":"a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c","rights_status":"source-license-verified","metadata_status":"verified","derivation":"source-svg-copy","viewBox":[0,0,191.25234,244.474996],"used_in":["00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer.md"]},
  {"asset":"00 Учебник/Assets/Figures/curated/seq2seq-attention-bridge-2026-11-10/cho-2014-figure4.png","sha256":"c597003c56432847b54db6fabec8d1da7de043adc96e367aa2a58fbf090a7e5a","source_url":"https://aclanthology.org/W14-4012/","source_image_url":"https://aclanthology.org/W14-4012.pdf#page=7","source_asset":"05 Источники/Originals/seq2seq-attention-bridge-2026-11-10/cho-w14-4012.pdf","source_asset_sha256":"f86daf916fcd28b24e8493545925421b925a7b60ee0a73c2e1e72f634a580226","license":"CC BY-NC-SA 3.0","license_url":"https://creativecommons.org/licenses/by-nc-sa/3.0/","license_evidence":"05 Источники/Originals/architecture-patterns-2026-10-21/acl-copyright.html","license_evidence_sha256":"88d644d2e21542ef1cc6c304c264ce7543d81e87c9599aff7e1c68e7a6df9558","rights_status":"source-license-verified","metadata_status":"verified","derivation":"pdf-page-render-crop","crop_box_px":[335,325,2595,935],"render_dpi":360,"source_page":7,"used_in":["00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора.md"]},
] as const;

const multiheadOriginals = [
  {"asset":"00 Учебник/Assets/Figures/curated/multihead-2026-11-10/vaswani-multihead-figure2.png","sha256":"ba8a3ede5d40183ce8c18de249be60bec692646d53db108a3df51797a3c8634a","source_url":"https://arxiv.org/html/1706.03762v7#S3.F2.g2","source_image_url":"https://arxiv.org/html/1706.03762v7/Figures/ModalNet-20.png","source_asset":"05 Источники/Originals/multihead-2026-11-10/ModalNet-20.png","source_asset_sha256":"ba8a3ede5d40183ce8c18de249be60bec692646d53db108a3df51797a3c8634a","source_page_snapshot":"05 Источники/Originals/transformer-gpt-bridge-2026-11-10/vaswani-v7.html","source_page_sha256":"fe1288e9c6f8322156429ab44bf5ced203155f8ecc7ea9fe6cfd125872c72a3b","license":"Google permission for attributed scholarly or journalistic figure reuse","license_url":"https://arxiv.org/html/1706.03762v7#p1.1","license_evidence":"05 Источники/Originals/transformer-gpt-bridge-2026-11-10/vaswani-v7.html","license_evidence_sha256":"fe1288e9c6f8322156429ab44bf5ced203155f8ecc7ea9fe6cfd125872c72a3b","derivation":"source-png-copy","metadata_status":"verified","rights_status":"source-license-verified","dimensions_px":[835,1282],"used_in":["00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров.md","en/00 Textbook/05 Attention and Transformer/03 Masking, multi-head attention, and tensor shapes.md"]},
  {"asset":"00 Учебник/Assets/Figures/curated/multihead-2026-11-10/voita-trained-heads-figure1.png","sha256":"27e4f18a5ebe6f806b82025c21b63278aa89a899fef61141e31d783cca86ee25","source_url":"https://aclanthology.org/P19-1580/","source_image_url":"https://aclanthology.org/P19-1580.pdf#page=3","source_asset":"05 Источники/Originals/multihead-2026-11-10/P19-1580.pdf","source_asset_sha256":"9726037ea63f0d500e724fbf9d387bc46c71b5f50626d63538f1172f132c0827","license":"CC BY 4.0","license_url":"https://creativecommons.org/licenses/by/4.0/","license_evidence":"05 Источники/Originals/architecture-patterns-2026-10-21/acl-copyright.html","license_evidence_sha256":"88d644d2e21542ef1cc6c304c264ce7543d81e87c9599aff7e1c68e7a6df9558","derivation":"pdf-page-render-crop","metadata_status":"verified","rights_status":"source-license-verified","dimensions_px":[2110,570],"source_page":3,"render_dpi":360,"crop_box_px":[450,300,2560,870],"used_in":["00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров.md","en/00 Textbook/05 Attention and Transformer/03 Masking, multi-head attention, and tensor shapes.md"]},
] as const;

const foundationsFigurePath = '00 Учебник/Assets/Figures/curated/autoencoder-dropout-primary-2026-11-10/';
const princeSourcePath = '05 Источники/Originals/autoencoder-dropout-primary-2026-11-10/';
const princeCommit = '4caea9b63ee7217ac6568e62394fc6c179fc91ac';
const princeFigures = {
  'prince-dropout-figure9-9.png': [9, 163, 'RegDropout.pdf', 609,
    '3292c1d01a093336f45ee273c186c034dc6f708bdd06b5b743b0ac51edf0abd4',
    '0b62f33819a80d7ce2a6542b5b177a52d90003cdc8ac6061c9ea8fd0f64e32f7'],
  'prince-vae-posterior-figure17-8.png': [17, 351, 'VAEVariational.pdf', 719,
    '391c9e2cc3dd55adff8ea686fca69e239a490c73fea5bf946a433cafc59b0591',
    'f4058a4208cc3fb23c342e0c5f84be10ce1a8ddf252277318a1772dc2a1d8479'],
  'prince-vae-reparameterization-figure17-11.png': [17, 353, 'VAEReparam.pdf', 613,
    '297b43063daceb35ece2be386230cac43236f441653ebf7c7228c1cd744ad5f7',
    'b2b33aa33e16135bf77e89374250ff4536316cac82b92a062f7862aa68c17854'],
} as const;

function validateFoundationsFigure(entry: CuratedAsset, repositoryRoot: string): void {
  const name = entry.asset.slice(foundationsFigurePath.length);
  const target = resolve(repositoryRoot, entry.asset);
  expect(entry.metadata_status).toBe('verified');
  expect(entry.rights_status).toBe('source-license-verified');
  expect(entry.provenance_confirmation).toBe('source-license-verified');
  expect(entry.crop_box_px).toBeUndefined();
  expect(entry.source_member).toBeUndefined();
  if (name === 'chervinskii-autoencoder.png') {
    expect(entry.used_in).toEqual([
      '00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE.md',
      '05 Источники/Courses/Harvard ML Systems/vol2/robust_ai.md',
    ]);
    expect(entry).toMatchObject({
      author: 'Chervinskii', derivation: 'source-raster-copy', source_asset: entry.asset,
      source_url: 'https://commons.wikimedia.org/wiki/File:Autoencoder_structure.png',
      source_image_url: 'https://upload.wikimedia.org/wikipedia/commons/2/28/Autoencoder_structure.png',
      license_evidence_url: 'https://commons.wikimedia.org/w/index.php?title=File:Autoencoder_structure.png&oldid=808656750',
      license: 'CC BY-SA 4.0', license_url: 'https://creativecommons.org/licenses/by-sa/4.0/',
      dimensions_px: [677, 506], source_asset_sha256: entry.sha256,
      sha256: '36bda1536cc36e02b56ee438834566fb659525e0752aa13377f5d4ea8f65f9af',
    });
    expect(createHash('sha1').update(readFileSync(target)).digest('hex')).toBe('0926f68e96701ef3edf0e4c577bea1a36e366cbb');
  } else {
    expect(entry.used_in).toHaveLength(1);
    expect(Object.hasOwn(princeFigures, name)).toBe(true);
    const [chapter, page, member, height, sourceHash, outputHash] = princeFigures[name as keyof typeof princeFigures];
    expect(entry).toMatchObject({
      author: 'Simon J. D. Prince', copyright: 'MIT Press', commit: princeCommit,
      source_url: 'https://udlbook.github.io/udlbook/',
      figure_url: `https://github.com/udlbook/udlbook/releases/download/v5.0.3/UnderstandingDeepLearning_02_09_26_C.pdf#page=${page}`,
      source_image_url: `https://raw.githubusercontent.com/udlbook/udlbook/${princeCommit}/PDFFigures/UDLChap${chapter}PDF.zip`,
      source_archive_member: member,
      source_archive_sha256: chapter === 9
        ? '037d9d0a84fedc556c6dc3d178c6513c09bc23cec6087b293584202d2fae5e5d'
        : '498f254ee41053f80593d6527a61a14ae707fc34e5906987e2c730847032d03b',
      source_asset: princeSourcePath + member, source_asset_sha256: sourceHash,
      source_page: 1, derivation: 'pdf-page-render', sha256: outputHash,
      render_scale_to_px: 1900, render_page_size_px: [1900, height], dimensions_px: [1900, height],
      license: 'CC BY-NC-ND 4.0', license_url: 'https://creativecommons.org/licenses/by-nc-nd/4.0/',
      license_evidence: princeSourcePath + 'udl-figures-LICENSE.txt',
      license_evidence_url: `https://github.com/udlbook/udlbook/blob/${princeCommit}/LICENSE`,
      license_evidence_sha256: '162cd1bd77ae1b548c505840fa24c0926a886a97f37fc6a14eb40a8e115b5eb9',
    });
    // Public checkouts have the pinned renders; the ignored source copies
    // allow an additional full-page pixel check when they are available.
    const pdf = resolve(repositoryRoot, entry.source_asset!);
    if (existsSync(pdf)) {
      expect(sha256(pdf)).toBe(sourceHash);
      const rendered = execFileSync(process.env.PDFTOPPM ?? 'pdftoppm', [
        '-f', '1', '-l', '1', '-scale-to', '1900', '-singlefile', '-png', pdf,
      ], { env: { ...process.env, LC_ALL: 'C', LANG: 'C' }, maxBuffer: 1024 * 1024 });
      expect(readFileSync(target)).toEqual(rendered);
    }
    const policy = resolve(repositoryRoot, entry.license_evidence!);
    if (existsSync(policy)) expect(sha256(policy)).toBe(entry.license_evidence_sha256);
  }
  expect(sha256(target)).toBe(entry.sha256);
  expect(pngDimensions(target)).toEqual(entry.dimensions_px);
  expectDecodableImage(target);
}

const archivedAgentPaperFigures = [
   {
     "asset": "00 Учебник/Assets/Figures/curated/astra/reward-ablation.png",
     "owner": "ASTRA",
     "figure_url": "https://arxiv.org/html/2601.21558v2#S4.F7",
     "sha256": "6dd44813a19c201bdfcc24b3786f79319659785afc8f67eb32b9f64338761899"
   },
   {
     "asset": "00 Учебник/Assets/Figures/curated/astra/rl-pipeline.png",
     "owner": "ASTRA",
     "figure_url": "https://arxiv.org/html/2601.21558v2#S3.F4",
     "sha256": "4df3e411875831c94bad8f2f04b8fadb2254012c68caef34fd67c848ab47ec37"
   },
   {
     "asset": "00 Учебник/Assets/Figures/curated/astra/tool-mixing-ablation.png",
     "owner": "ASTRA",
     "figure_url": "https://arxiv.org/html/2601.21558v2#S4.F6",
     "sha256": "fa1e85bfdcb100d58f079d5e1618cf381e5b511baa5f80d140b0f0cfa59f03ab"
   },
   {
     "asset": "00 Учебник/Assets/Figures/curated/astra/turn-distribution.png",
     "owner": "ASTRA",
     "figure_url": "https://arxiv.org/html/2601.21558v2#S4.F8",
     "sha256": "ece15cda113d0b95d195375341ada78096a6da7b46e5d93517384a3db844e480"
   },
   {
     "asset": "00 Учебник/Assets/Figures/curated/toolrl/cold-start-comparison.png",
     "owner": "ToolRL",
     "figure_url": "https://arxiv.org/html/2504.13958v1#S5.F11",
     "sha256": "99fcb5fe026f97224fb6e06109d67ea5a7687005691b3d2b25fc9bd5ec6a82df"
   },
   {
     "asset": "00 Учебник/Assets/Figures/curated/toolrl/introduction.png",
     "owner": "ToolRL",
     "figure_url": "https://arxiv.org/html/2504.13958v1#S1.F1",
     "sha256": "7c80d3359f7dc2903ca0255bf12b1e33ebf6099fca17c24b79b02dbb7a7e3993"
   }
 ] as const;

const finalSourceFigures = [
   {
     "author": "TensorFlow documentation authors",
     "title": "Word embeddings: One-hot encodings",
     "source_url": "https://www.tensorflow.org/text/guide/word_embeddings#one-hot_encodings",
     "source_image_url": "https://www.tensorflow.org/static/text/guide/images/one-hot.png",
     "license_evidence_url": "https://www.tensorflow.org/text/guide/word_embeddings",
     "source_page_snapshot": "05 Источники/Originals/reviewed-source-figures-2026-11-10/tensorflow-guide.html",
     "source_page_sha256": "cbc5b9594bb53b2cc1aedb66d41bdf11e51e276e13e51f4823dc6290a4938cc7",
     "sha256": "0309948602d54e32de834032a266fe02179f5a6432bb6aaf0dd396cf9084373a",
     "dimensions_px": [
       638,
       522
     ],
     "license": "CC BY 4.0",
     "license_url": "https://creativecommons.org/licenses/by/4.0/",
     "derivation": "source-raster-copy",
     "used_in": [
       "00 Учебник/02 Представление текста и токенизация/01 Представление текста числами.md"
     ],
     "asset": "00 Учебник/Assets/Figures/curated/reviewed-source-figures-2026-11-10/tensorflow-one-hot.png",
     "source_asset": "00 Учебник/Assets/Figures/curated/reviewed-source-figures-2026-11-10/tensorflow-one-hot.png",
     "source_asset_sha256": "0309948602d54e32de834032a266fe02179f5a6432bb6aaf0dd396cf9084373a",
     "metadata_status": "verified",
     "provenance_confirmation": "source-license-verified",
     "rights_status": "source-license-verified"
   },
   {
     "author": "Mercurysheet, adapted from Cmglee",
     "title": "Reduced Singular Value Decompositions",
     "source_url": "https://commons.wikimedia.org/wiki/File:Reduced_Singular_Value_Decompositions.svg",
     "source_image_url": "https://upload.wikimedia.org/wikipedia/commons/c/c4/Reduced_Singular_Value_Decompositions.svg",
     "license_evidence_url": "https://commons.wikimedia.org/w/index.php?title=File:Reduced_Singular_Value_Decompositions.svg&oldid=1024634772",
     "source_page_snapshot": "05 Источники/Originals/reviewed-source-figures-2026-11-10/commons-svd.html",
     "source_page_sha256": "54f52b925f53d3a242135ffdc5efaaed043c8b5576b720bb153452cac266f3bb",
     "sha256": "f170f19edaaa8de49b840393e548e1ecb70d2eb4be7ca6f525a737c61065eec8",
     "license": "CC BY-SA 4.0",
     "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
     "derivation": "source-svg-copy",
     "used_in": [
       "00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings.md"
     ],
     "asset": "00 Учебник/Assets/Figures/curated/reviewed-source-figures-2026-11-10/reduced-svd.svg",
     "source_asset": "00 Учебник/Assets/Figures/curated/reviewed-source-figures-2026-11-10/reduced-svd.svg",
     "source_asset_sha256": "f170f19edaaa8de49b840393e548e1ecb70d2eb4be7ca6f525a737c61065eec8",
     "metadata_status": "verified",
     "provenance_confirmation": "source-license-verified",
     "rights_status": "source-license-verified"
   },
   {
     "author": "Nathan Lambert",
     "title": "Training Overview, Figure 5: early RLHF",
     "source_url": "https://github.com/natolambert/rlhf-book/blob/3f573487c58548c16a548cd5fb8979a0189384c7/book/images/rlhf-basic.png",
     "source_image_url": "https://raw.githubusercontent.com/natolambert/rlhf-book/3f573487c58548c16a548cd5fb8979a0189384c7/book/images/rlhf-basic.png",
     "figure_url": "https://rlhfbook.com/c/03-training-overview#fig:rlhf-basic-repeat",
     "license_evidence_url": "https://github.com/natolambert/rlhf-book/blob/3f573487c58548c16a548cd5fb8979a0189384c7/README.md#license",
     "source_page_snapshot": "05 Источники/Originals/reviewed-source-figures-2026-11-10/lambert-chapter.md",
     "source_page_sha256": "bbf587036cf9931038fc00d4928021dad05abaaa482fd31b8625e04495543f83",
     "license_evidence": "05 Источники/Originals/reviewed-source-figures-2026-11-10/lambert-readme.md",
     "license_evidence_sha256": "2faf557d3cbd6d114c6e29112f50dc01abdb0a0def136dde34b021f50d37b0f9",
     "license_notice": "site/public/licenses/rlhf-book-MIT.txt",
     "license_notice_sha256": "a312d5db1e63cf298985a9989069ce7d6a7a35ac237572276ab3222e5af42d4d",
     "commit": "3f573487c58548c16a548cd5fb8979a0189384c7",
     "sha256": "67705f0e237d4eff26e2b1b49e16fdd6aac007aa986336f661ef9cb04eb906cc",
     "dimensions_px": [
       4018,
       1185
     ],
     "license": "MIT",
     "license_url": "https://github.com/natolambert/rlhf-book/blob/3f573487c58548c16a548cd5fb8979a0189384c7/LICENSE-CODE",
     "derivation": "source-raster-copy",
     "used_in": [
       "00 Учебник/12 Post-training и Alignment/01 SFT и instruction data.md",
       "03 Исследовательские линии/RLHF → DPO → RLVR.md",
       "Concepts/Training/RLHF.md"
     ],
     "asset": "00 Учебник/Assets/Figures/curated/reviewed-source-figures-2026-11-10/rlhf-basic.png",
     "source_asset": "00 Учебник/Assets/Figures/curated/reviewed-source-figures-2026-11-10/rlhf-basic.png",
     "source_asset_sha256": "67705f0e237d4eff26e2b1b49e16fdd6aac007aa986336f661ef9cb04eb906cc",
     "metadata_status": "verified",
     "provenance_confirmation": "source-license-verified",
     "rights_status": "source-license-verified"
   }
 ] as const;
const finalRetiredFigures = [
   {
     "asset": "00 Учебник/Assets/Figures/curated/instructgpt/chatgpt-training-pipeline.png",
     "figure_url": "https://huyenchip.com/2023/05/02/rlhf.html#rlhf-overview",
     "sha256": "ffb549dc81ca08e62ae8d936d2313f41711ef5def80e834e89ced8926f4c8724",
     "former_used_in": [
       "00 Учебник/12 Post-training и Alignment/01 SFT и instruction data.md",
       "Concepts/Training/RLHF.md",
       "03 Исследовательские линии/RLHF → DPO → RLVR.md"
     ],
     "replacement": "00 Учебник/Assets/Figures/curated/reviewed-source-figures-2026-11-10/rlhf-basic.png"
   },
   {
     "asset": "00 Учебник/Assets/Figures/curated/voita-word-embeddings/one-hot.png",
     "figure_url": "https://lena-voita.github.io/nlp_course/word_embeddings.html#one_hot_vectors",
     "sha256": "b1284449dfd787afc529b313bb3f95d08b5dfa8649034a57e370efc8a7239272",
     "former_used_in": [
       "00 Учебник/02 Представление текста и токенизация/01 Представление текста числами.md"
     ],
     "replacement": "00 Учебник/Assets/Figures/curated/reviewed-source-figures-2026-11-10/tensorflow-one-hot.png"
   },
   {
     "asset": "00 Учебник/Assets/Figures/curated/voita-word-embeddings/count-based-idea.png",
     "figure_url": "https://lena-voita.github.io/nlp_course/word_embeddings.html#pre_neural",
     "sha256": "bdd6d2fcd48116d0fe055e6fceaa051a4b95128c4367d5c591102f5ee0dcbd05",
     "former_used_in": [
       "00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings.md"
     ],
     "replacement": "00 Учебник/Assets/Figures/curated/reviewed-source-figures-2026-11-10/reduced-svd.svg"
   }
 ] as const;

function validateFinalSourceFigure(entry: CuratedAsset, repositoryRoot: string): boolean {
  const expected = finalSourceFigures.find(row => row.asset === entry.asset);
  if (!expected) return false;
  expect(entry).toMatchObject(expected);
  expect(entry.crop_box_px).toBeUndefined();
  const file = resolve(repositoryRoot, entry.asset);
  expect(sha256(file)).toBe(expected.sha256);
  expectDecodableImage(file);
  if ('dimensions_px' in expected) expect(pngDimensions(file)).toEqual(expected.dimensions_px);
  const snapshot = resolve(repositoryRoot, expected.source_page_snapshot);
  if (existsSync(snapshot)) expect(sha256(snapshot)).toBe(expected.source_page_sha256);
  if (entry.asset.endsWith('/reduced-svd.svg')) {
    const bytes = readFileSync(file);
    expect(createHash('sha1').update(bytes).digest('hex')).toBe('09d791ce9350505608a65ddfa4671eb8d0b8a8ba');
    expect(bytes.toString('utf8')).not.toMatch(/<script\b|\b(?:xlink:)?href=["'](?!#)/i);
  }
  if ('license_notice' in expected) {
    expect(sha256(resolve(repositoryRoot, expected.license_notice))).toBe(expected.license_notice_sha256);
    const evidence = resolve(repositoryRoot, expected.license_evidence);
    if (existsSync(evidence)) {
      expect(sha256(evidence)).toBe(expected.license_evidence_sha256);
      expect(readFileSync(evidence, 'utf8')).toContain('some images in `book/images/` are unlicensed photos or screenshots');
    }
  }
  return true;
}

const retiredFigureRightsReview = [
  ["atlas-courses-official/llama3-architecture.png","eb052d751a0112841986dd77ed3ff5c6c2b6478210548735c29b884386a02db8","paper-figure-license-unverified"],
  ["source-first-65-68/react-patterns.png","d5754118adcab0d07293244257d15f6cbc4558610329be370d1a38972d848d2f","project-figure-license-unverified"],
  ["stanford-cs336-2026/architectures/qwen-next-hybrid-architecture.jpg","3c277992cb4615d4a55ac55ffb328af633e4d597e51f90019103ce670cb122ad","third-party-figure-license-unverified"],
  ["alternatives-transformer-2026/berkeley-bahdanau-attention.png","0f7dcd9e44c7de2fbaa140865f0187501269c461a3d33b638fb87bb204b609d4","course-figure-license-unverified"],
  ["alternatives-transformer-2026/berkeley-qwen3-next-gdn.png","59568094fe677afd653bb0a87b3bf31f98ad20cb5251ecd28860e9bcd196585b","course-figure-license-unverified"],
  ["alternatives-transformer-2026/berkeley-seq2seq-bottleneck.png","bd6343369f276d03c52db5929c7fdff00a5cc379f220c91348f56148ee563ace","course-figure-license-unverified"],
  ["alternatives-transformer-2026/berkeley-transformer-language-model.png","8b990ac363185fbc43608da8b96870fa495eaf6a81107e4f45a9c5ae1c0b24b5","course-figure-license-unverified"],
  ["alternatives-transformer-2026/cmu-hybrid-models.png","0f6632a6d1cc4905b2ab81f495a145514b147fb9ccf54fae5953243d4c788bc9","course-figure-license-unverified"],
  ["alternatives-transformer-2026/cmu-linear-attention.png","b38daa75f7ae1fcc2ebdc10f2660275ccc69797bf0e8bdbd8c73e8c78e678c95","course-figure-license-unverified"],
  ["alternatives-transformer-2026/mamba-selective-scan-course.png","d7c5864a4f82f22781a756d203bdceec64852fae3e883dc30199c20d6a3f2ee6","course-figure-license-unverified"],
  ["alternatives-transformer-2026/stanford-cs230-lstm.png","0d785e115dbdda9e33a1ec2dbf0f6951ab125811d1b74a6de22b0a17de7f6021","course-figure-license-unverified"],
  ["alternatives-transformer-2026/stanford-cs230-rnn-unrolled.png","8683c721faf9012f6b07cc84adc5242e4ff4723cb30b2f3bdf140562c234bb25","course-figure-license-unverified"],
] as const;

const d2lLegacyOriginals = [
  ["d2l/lenet.svg","66dc7de32432aebb461dd1845fae53880f07328c03040009f55dd4a0cacbbd11","https://raw.githubusercontent.com/d2l-ai/d2l-en/b2e2ae30898a9d0126a9699ae7e441de3e272715/img/lenet-vert.svg"],
  ["d2l/resnet18.svg","2ffaf03da85439f7796dfe5f0ff6113479bdd8c5248740fdf7ca7fa37ffe7883","https://raw.githubusercontent.com/d2l-ai/d2l-en/b2e2ae30898a9d0126a9699ae7e441de3e272715/img/resnet18.svg"],
  ["source-first-7-14/d2l-sigmoid.svg","919f4d5fe95251b8a8337fb90f5e4c5869eb58cccdf890af3bba5afc2c072776","/_images/output_mlp_76f463_48_0.svg"],
  ["source-first-7-14/d2l-sigmoid-gradient.svg","308927d0b768f063a7760142d0233be44090340f8dc9b916a1d36409dc548368","/_images/output_mlp_76f463_63_0.svg"],
  ["source-first-7-14/d2l-tanh.svg","b31896fdf836337c92d40d9506c29d9a7129f4221b4e9bf9531201ddf9572b0f","/_images/output_mlp_76f463_78_0.svg"],
  ["source-first-7-14/d2l-tanh-gradient.svg","a0b8373b2d090121895b6a85dcd43d6af0afab3c2d7ae52bb3bbf0e0f03c29db","/_images/output_mlp_76f463_93_0.svg"],
  ["source-first-7-14/d2l-relu.svg","350dfd434b6c4210d608a1d55f58a69f07ee28958a14978d17495e60fb59c53c","/_images/output_mlp_76f463_18_0.svg"],
  ["source-first-7-14/d2l-relu-gradient.svg","1c56cdc4b9c5092bf165e96df001cda87a432b60a9e24494a3d5675bf15b7810","/_images/output_mlp_76f463_33_0.svg"],
  ["source-first-7-14/d2l-sgd-anisotropic.svg","d3dc77b38a500a85574e5df17e800dd08cd938a23ced8311046229a7f9d5676c","/_images/output_momentum_e3683f_3_1.svg"],
  ["source-first-7-14/d2l-momentum-anisotropic.svg","8e68f49a0ffdaa64da9daeabc2065e48476154d5fdb8d79a1a3f97d39ac291cc","/_images/output_momentum_e3683f_27_1.svg"],
  ["d2l/dropout.svg","ad463729d07aa97f9f48d136b2aa205504444720eea02cdb9b8a94eef145aa19","/_images/dropout2.svg"],
  ["source-first-7-14/d2l-cross-correlation.svg","a29e9c6b51db35de2e2c68e60177eaa116050ea11c397e0f8715cf64f6b898be","/_images/correlation.svg"],
] as const;

type ReviewedLegacySource = {
  asset: string;
  route: string;
  sha256: string;
  bytes: number;
  sourceImage: string;
  sourceAsset: string;
  license: string;
  licenseUrl: string;
  licenseEvidence: string;
  licenseEvidenceSha256?: string;
  publicNotice?: string;
  publicNoticeSha256?: string;
  commit?: string;
  dimensions?: number[];
  viewBox?: number[];
};
const reviewedLegacySources = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-10-legacy-textbook-source-copies-additions.json'), 'utf8')).verifiedOriginals as ReviewedLegacySource[];

const stableDiffusionAlammarFigures: Record<string, {
  sha256: string; blob: string; dimensions: number[]; section: string;
}> = {
  'stable-diffusion-text-dataset-v2.png': {
    sha256: 'c599107951905474b4aa6d3aaf9b96c8499f51c0918021d21f2648df4b6a0569',
    blob: '6bcadf1a27c304d34e1aa4adae5229b22e21b694',
    dimensions: [2352, 1406],
    section: 'feeding-text-information-into-the-image-generation-process',
  },
  'stable-diffusion-components-and-tensors.png': {
    sha256: '329cc9d2251837e26578f1330aee55feba5bd2307934105a48ca7eaee9ee6b1e',
    blob: '43d349e75e7c6001ab6fcaaa648d9526185ab4e9',
    dimensions: [1440, 553],
    section: 'the-components-of-stable-diffusion',
  },
};

const legacyPdfFigureEvidence = (JSON.parse(readFileSync(resolve(
  root, 'docs/audit/2026-11-11-legacy-pdf-figure-evidence-additions.json'
), 'utf8')) as { proofs: { record: CuratedAsset; comparison: string }[] }).proofs;

const retiredVlmFigures = (JSON.parse(readFileSync(resolve(
  root, 'docs/audit/2026-11-11-vlm-reader-figure-resolution-additions.json'
), 'utf8')) as { retired: CuratedAsset[] }).retired;

type ServingRepositoryOriginal = {
  asset: string; sha256: string; source_asset: string; source_asset_sha256: string;
  git_commit: string; git_blob: string; upstream_url: string; license_url: string;
  license_file: string; license_sha256: string;
};
const servingRepositoryReview = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-mooncake-dynamo-originals-additions.json'), 'utf8')) as {
    figures: ServingRepositoryOriginal[];
    archivedDynamoOverview: ServingRepositoryOriginal;
  };
const servingRepositoryOriginals = [
  ...servingRepositoryReview.figures, servingRepositoryReview.archivedDynamoOverview,
];

const sequenceFigureReview = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-sequence-architecture-figures-additions.json'), 'utf8')) as {
    figures: CuratedAsset[];
    historical: (CuratedAsset & { replacement: string; former_used_in: string[] })[];
  };
const sequenceRecallReview = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-sequence-recall-licensed-figures-additions.json'), 'utf8')) as {
    figures: CuratedAsset[];
    retired: (CuratedAsset & { replacement: string; former_used_in: string[] })[];
  };
const optimizerCurvatureReview = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-optimizer-curvature-source-figures-additions.json'), 'utf8')) as {
    figures: CuratedAsset[];
    retired: CuratedAsset & { replacement: string; former_used_in: string[] };
  };
const optimizerCurvatureAssets = [...optimizerCurvatureReview.figures, optimizerCurvatureReview.retired];
const vllmSourceLibrary = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-vllm-source-library-figure-reconciliation-additions.json'), 'utf8')) as {
    figures: CuratedAsset[];
    retired: (CuratedAsset & { git_blob: string; bytes: number; former_used_in: string[] })[];
  };

const distributedNvidiaFigures = (JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-distributed-nvidia-source-figures-additions.json'), 'utf8')) as {
    figures: (CuratedAsset & {
      sha256: string; bytes: number; git_blob: string; dimensions_px: [number, number];
      source_asset: string; license_evidence: string; license_evidence_sha256: string;
    })[];
  }).figures;

const chenSourceFigures = (JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-chen-speculative-source-figures-additions.json'), 'utf8')) as {
    figures: (CuratedAsset & {
      source_asset: string; source_parent_asset: string; source_parent_sha256: string;
      sha256: string; git_blob: string; bytes: number; dimensions_px: [number, number];
      replacement?: string;
    })[];
  }).figures;

const flashSourceReview = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-flashattention-source-figures-additions.json'), 'utf8')) as {
    figures: CuratedAsset[];
    historical: (CuratedAsset & { replacement?: string })[];
  };
const gptRouteFigure = (JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-gpt-route-experimental-evidence-additions.json'), 'utf8')) as {
    figure: CuratedAsset;
  }).figure;
const reconciledReaderFigures = (JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-textbook-visible-figure-provenance-additions.json'), 'utf8')) as { records: CuratedAsset[] }).records;
const schedulingSourceReview = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-scheduling-originals-additions.json'), 'utf8')) as {
    figures: CuratedAsset[];
    retired: (CuratedAsset & { replacement: string })[];
  };
const lmcacheTransferReview = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-lmcache-transfer-source-figures-additions.json'), 'utf8')) as {
    figures: CuratedAsset[];
    retired: (CuratedAsset & { replacements: string[] })[];
  };
const inferenceTimelineReview = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-inference-roofline-latency-figures-additions.json'), 'utf8')) as {
    figure: CuratedAsset;
    historical: (CuratedAsset & { replacement: string })[];
  };
const quantSourceReview = JSON.parse(readFileSync(resolve(root,
  'docs/audit/2026-11-11-quantization-source-figures-additions.json'), 'utf8')) as {
    figures: CuratedAsset[];
    historical: (CuratedAsset & { replacements: string[] })[];
  };

function validateCuratedSourceAsset(entry: CuratedAsset, repositoryRoot = root): void {
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/distill-rlvr-2026-11-14/')) {
    const paper = 'https://arxiv.org/html/2506.14245v2';
    const originals = {
      'wen-figure6a.svg': ['S6.F6.sf1', 'wxm_comp_sft_data_pass1.svg', 'cc9b9f8cfe64c75936bb20b99ab979363fd1628eb9225a48c91a729bdc4a8241', [0, 0, 1085.1714, 1084.7913]],
      'wen-figure6b.svg': ['S6.F6.sf2', 'wxm_comp_sft_rlvr_passk.svg', '48cb10349731a10ef465b97f63d3be5f5e1349cd4ee46cc8510cb2ed8599bada', [0, 0, 1086.3856, 1085.6488]],
    } as const;
    const name = entry.asset.split('/').at(-1)! as keyof typeof originals;
    const [anchor, image, digest, viewBox] = originals[name];
    expect(entry).toMatchObject({
      author: 'Xumeng Wen et al.', source_url: paper + '#' + anchor,
      source_image_url: paper + '/' + image, rights_evidence: paper,
      license: 'CC BY 4.0', license_url: 'https://creativecommons.org/licenses/by/4.0/',
      derivation: 'source-svg-copy', source_asset: entry.asset, sha256: digest,
      source_asset_sha256: digest, viewBox, metadata_status: 'verified',
      rights_status: 'source-license-verified', provenance_confirmation: 'source-license-verified',
      used_in: ['00 Учебник/12 Post-training и Alignment/08 Reasoning distillation.md'],
    });
    expect(entry.crop_box_px).toBeUndefined();
    const file = resolve(repositoryRoot, entry.asset);
    expect(sha256(file)).toBe(digest);
    expect(readFileSync(file, 'utf8').split('\n')[0]).toContain(`viewBox="${viewBox.join(' ')}"`);
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/grpo-dapo-2026-11-14/wen-figure4.svg') {
    const paper = 'https://arxiv.org/html/2506.14245v2';
    const digest = '98db9ba58301c872a6dfbb5d01b4117e884ecbe870a2544d584ec3ccfef70c33';
    expect(entry).toMatchObject({
      author: 'Xumeng Wen et al.', source_url: paper + '#S5.F4',
      source_image_url: paper + '/rollout_verify_easy.svg', rights_evidence: paper,
      license: 'CC BY 4.0', license_url: 'https://creativecommons.org/licenses/by/4.0/',
      derivation: 'source-svg-copy', source_asset: entry.asset, sha256: digest,
      source_asset_sha256: digest, viewBox: [0, 0, 1040.2282, 309.38648],
      metadata_status: 'verified', rights_status: 'source-license-verified',
      provenance_confirmation: 'source-license-verified',
      used_in: ['00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1.md'],
    });
    expect(entry.crop_box_px).toBeUndefined();
    const file = resolve(repositoryRoot, entry.asset);
    expect(sha256(file)).toBe(digest);
    expect(readFileSync(file, 'utf8').split('\n')[0]).toContain('viewBox="0 0 1040.2282 309.38648"');
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/rlvr-pass-k-2026-11-14/yue-figure-2.svg') {
    const paper = 'https://arxiv.org/html/2504.13837v5';
    const digest = '408381de34bfbf22e1119f58b395236629996be0ebb1648c4848b56730b873c2';
    expect(entry).toMatchObject({
      author: 'Yang Yue et al.', source_url: paper + '#S2.F2',
      source_image_url: paper + '/simpleRL_coverage_main.svg',
      rights_evidence: paper, license: 'CC BY 4.0',
      license_url: 'https://creativecommons.org/licenses/by/4.0/',
      derivation: 'source-svg-copy', source_asset: entry.asset,
      sha256: digest, source_asset_sha256: digest, viewBox: [0, 0, 3025.5, 2825.25],
      metadata_status: 'verified', rights_status: 'source-license-verified',
      provenance_confirmation: 'source-license-verified',
      used_in: ['00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers.md'],
    });
    expect(entry.crop_box_px).toBeUndefined();
    const file = resolve(repositoryRoot, entry.asset);
    expect(sha256(file)).toBe(digest);
    expect(readFileSync(file, 'utf8').split('\n')[0]).toContain('viewBox="0 0 3025.5 2825.25"');
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/dpo-ivison-2026-11-14/')) {
    const paper = 'https://arxiv.org/html/2406.09279v2';
    const evidence = '05 Источники/Originals/dpo-ivison-2026-11-14/ivison-v2.html';
    expect(entry).toMatchObject({
      author: 'Hamish Ivison et al.', license: 'CC BY 4.0',
      license_url: 'https://creativecommons.org/licenses/by/4.0/',
      rights_evidence: paper, license_evidence: evidence,
      license_evidence_sha256: 'c7c71b459385a4f8f154c5bb207d237f61666d4b6826b22cdb030a871d943b4d',
      metadata_status: 'verified', rights_status: 'source-license-verified',
      provenance_confirmation: 'source-license-verified',
      used_in: ['00 Учебник/12 Post-training и Alignment/05 DPO.md'],
    });
    const html = resolve(repositoryRoot, evidence);
    expect(sha256(html)).toBe(entry.license_evidence_sha256);
    expect(readFileSync(html, 'utf8')).toContain('License: CC BY 4.0');
    const path = resolve(repositoryRoot, entry.asset);
    if (entry.asset === '00 Учебник/Assets/Figures/curated/dpo-ivison-2026-11-14/paired-results.png') {
      expect(entry).toMatchObject({
        source_url: paper + '#S3.T2',
        source_image_url: 'https://arxiv.org/pdf/2406.09279v2#page=5',
        source_asset: '05 Источники/Originals/dpo-ivison-2026-11-14/ivison-v2.pdf',
        source_asset_sha256: '5c49b1fa8f90514e9c740da81129a4344b363d8744fbef8bfc48f08777146237',
        sha256: '55b8b339b4c7355e898713cc555d413140605f5627e1b26838fb5474b802fdd8', derivation: 'pdf-page-render-crop',
        source_page: 5, render_dpi: 260, crop_box_px: [372, 1094, 1862, 1732],
        dimensions_px: [1490, 638],
      });
      expect(validateRenderedPngBytes(readFileSync(path))).toEqual([1490, 638]);
    } else {
      expect(entry.asset).toBe('00 Учебник/Assets/Figures/curated/dpo-ivison-2026-11-14/math-prompts.svg');
      expect(entry).toMatchObject({
        source_url: paper + '#S3.F3', source_image_url: paper + '/math_prompts.svg',
        source_asset: entry.asset,
        sha256: '790ebf009562bff524e4a781349d30ddf2853f47ac9174af7221f05b1a1aba87',
        source_asset_sha256: '790ebf009562bff524e4a781349d30ddf2853f47ac9174af7221f05b1a1aba87',
        derivation: 'source-svg-copy', viewBox: [0,0,576,432],
      });
      expect(readFileSync(path, 'utf8').slice(0, 200)).toContain('viewBox="0 0 576 432"');
    }
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(path)).toBe(entry.sha256);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/cai-evaluation-2026-11-14/')) {
    const paper = 'https://arxiv.org/html/2212.08073v1';
    const evidence = '05 Источники/Originals/training-consolidation-2026-10-29/cai-arxiv-v1.html';
    const pinned = {
      'human-elo-frontier.png': ['S1.F2', 'Constitutional.png', '9fd352f1052c8b0120cc32838c63b00057cc80f766b84509e59a3da36204f353'],
      'absolute-harmfulness.svg': ['S4.F10', 'HScore.svg', '313cdeac089920b6fb62109e5f99c481ac6b28d6d6dcc940e6df0eb6b9df9ce3'],
    } as const;
    const name = entry.asset.split('/').at(-1)! as keyof typeof pinned;
    expect(pinned[name]).toBeDefined();
    const [anchor, original, digest] = pinned[name];
    expect(entry).toMatchObject({
      author: 'Yuntao Bai et al.', source_url: paper + '#' + anchor,
      source_image_url: paper + '/' + original, source_asset: entry.asset,
      rights_evidence: paper, license: 'CC BY 4.0',
      license_url: 'https://creativecommons.org/licenses/by/4.0/',
      license_evidence: evidence,
      license_evidence_sha256: '2f175dac8902db65a5c0a7cd280109fa91e0d3c1d129e49edb9ab439edc3c97d',
      sha256: digest, source_asset_sha256: digest,
      metadata_status: 'verified', provenance_confirmation: 'source-license-verified',
      rights_status: 'source-license-verified', used_in: ['00 Учебник/12 Post-training и Alignment/00a Цели alignment — HHH, Constitutional AI и RLAIF.md'],
    });
    expect(sha256(resolve(repositoryRoot, evidence))).toBe(entry.license_evidence_sha256);
    expect(readFileSync(resolve(repositoryRoot, evidence), 'utf8')).toContain('creativecommons.org/licenses/by/4.0/');
    expect(entry.crop_box_px).toBeUndefined();
    const bytes = readFileSync(resolve(repositoryRoot, entry.asset));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(digest);
    if (name.endsWith('.png')) {
      expect(entry).toMatchObject({ derivation: 'source-png-copy', dimensions_px: [1140, 652] });
      expect(validateRenderedPngBytes(bytes)).toEqual([1140, 652]);
    } else {
      expect(entry).toMatchObject({ derivation: 'source-svg-copy', viewBox: [0, 0, 385.77815, 266.50376] });
      expect(bytes.toString().slice(0, 250)).toContain('viewBox="0 0 385.77815 266.50376"');
    }
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/ppo-areal-2026-11-14/areal-interruptible-timeline.png') {
    const edition = 'https://arxiv.org/html/2505.24298v5';
    const digest = '5744061bb43d2cf1a0da26864a1791c72c2754f9f6aa995fc415a9af71a920e4';
    expect(entry).toMatchObject({
      author: 'Wei Fu et al.', source_url: edition + '#S4.F3',
      source_image_url: edition + '/figures/interruptible-gen-timeline.png',
      rights_evidence: edition, license: 'CC BY-NC-ND 4.0',
      license_url: 'https://creativecommons.org/licenses/by-nc-nd/4.0/',
      derivation: 'source-png-copy', source_asset: entry.asset,
      sha256: digest, source_asset_sha256: digest, dimensions_px: [3332, 639],
      metadata_status: 'verified', rights_status: 'source-license-verified',
      provenance_confirmation: 'source-license-verified',
      used_in: ['00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM.md'],
    });
    expect(entry.crop_box_px).toBeUndefined();
    const path = resolve(repositoryRoot, entry.asset);
    expect(sha256(path)).toBe(digest);
    expect(validateRenderedPngBytes(readFileSync(path))).toEqual([3332, 639]);
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/gpt-route-2026-11-14/cs224n-vocabulary-head.png') {
    const course = 'https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/';
    const slide = course + 'slides/cs224n-2023-lecture9-pretraining.pdf#page=42';
    expect(entry).toMatchObject({
      author: 'John Hewitt, Stanford CS224N',
      license: 'Stanford CS224N course resource reuse notice',
      license_url: course + '#logistics', license_evidence_url: course + '#logistics',
      source_url: slide, source_image_url: slide,
      source_asset: '05 Источники/Originals/t5-2026-10-22/cs224n-2023-pretraining.pdf',
      source_asset_sha256: 'e29d1700e74468e811000a2ae2d8aadc63b1b0638d553dbc08102d0e3ff0e771',
      sha256: 'f9a668f5be174151f5d2464540ca58f4f85217c01687e62d949ae93862246154',
      source_page: 42, render_dpi: 240, crop_box_px: [1920, 612, 3012, 1402],
      dimensions_px: [1092, 790], derivation: 'pdf-page-render-crop',
      metadata_status: 'verified', rights_status: 'source-permission-verified',
      provenance_confirmation: 'cs224n-2023-course-permission',
      used_in: ['00 Учебник/_index.md', 'en/00 Textbook/_index.md'],
    });
    const path = resolve(repositoryRoot, entry.asset);
    expect(sha256(path)).toBe(entry.sha256);
    expect(pngDimensions(path)).toEqual(entry.dimensions_px);
    const source = resolve(repositoryRoot, entry.source_asset);
    if (existsSync(source)) expect(sha256(source)).toBe(entry.source_asset_sha256);
    for (const owner of entry.used_in!) {
      const body = readFileSync(resolve(repositoryRoot, owner), 'utf8');
      expect(body).toContain(entry.asset + '|');
      expect(body).toContain(slide);
      expect(body).toContain(course + '#logistics');
    }
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/gpt1-pretraining-2026-11-10/cs224n-decoder-classifier.png') {
    const course = 'https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/';
    const slide = course + 'slides/cs224n-2023-lecture9-pretraining.pdf#page=41';
    expect(entry).toMatchObject({
      author: 'Christopher Manning, Stanford CS224N',
      license: 'Stanford CS224N course resource reuse notice',
      license_url: course + '#logistics', license_evidence_url: course + '#logistics',
      source_url: slide, source_image_url: slide,
      source_asset: '05 Источники/Originals/t5-2026-10-22/cs224n-2023-pretraining.pdf',
      source_asset_sha256: 'e29d1700e74468e811000a2ae2d8aadc63b1b0638d553dbc08102d0e3ff0e771',
      sha256: 'c68a1e607ec6d776d8070a1025ed768c60c5cfeeb3b71ace3d230a9a626a196f',
      source_page: 41, render_dpi: 240, crop_box_px: [1835, 399, 2910, 1319],
      dimensions_px: [1075, 920], derivation: 'pdf-page-render-crop',
      metadata_status: 'verified', rights_status: 'source-permission-verified',
      provenance_confirmation: 'cs224n-2023-course-permission',
      used_in: [
        '00 Учебник/_index.md',
        'en/00 Textbook/_index.md',
        '00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение.md',
        'en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/04 GPT-1 — generative pre-training.md',
      ],
    });
    const path = resolve(repositoryRoot, entry.asset);
    expect(sha256(path)).toBe(entry.sha256);
    expect(pngDimensions(path)).toEqual(entry.dimensions_px);
    const source = resolve(repositoryRoot, entry.source_asset);
    if (existsSync(source)) expect(sha256(source)).toBe(entry.source_asset_sha256);
    for (const owner of entry.used_in!) {
      const body = readFileSync(resolve(repositoryRoot, owner), 'utf8');
      expect(body).toContain(entry.asset + '|');
      expect(body).toContain(slide);
      expect(body).toContain(course + '#logistics');
    }
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/vlm-2026/qwen25omni-figure3-tmrope.png') {
    const archive = '05 Источники/Originals/omni-2026-10-20/';
    expect(entry).toMatchObject({
      license: 'CC BY 4.0', license_url: 'https://creativecommons.org/licenses/by/4.0/',
      provenance_confirmation: 'source-license-verified', rights_status: 'source-license-verified',
      figure_url: 'https://arxiv.org/pdf/2503.20215v1#page=4', source_page: 4,
      source_asset: archive + 'qwen25omni.pdf', derivation: 'pdf-page-render-crop',
      license_evidence: archive + 'qwen25omni-abstract.html',
      license_evidence_sha256: 'cc7374fb27c3e71bd8fe280483821873e8bfe0263100c895f8eabdd3cb8e9b5c',
      sha256: 'c0db3d7a142bc4e61ce52a93e55951c362b30cc06049252fd657db879b457d25',
      dimensions_px: [1745, 475], crop_box_px: [260, 350, 2005, 825],
    });
    const path = resolve(repositoryRoot, entry.asset);
    expect(sha256(path)).toBe(entry.sha256);
    expect(pngDimensions(path)).toEqual(entry.dimensions_px);
    const evidence = resolve(repositoryRoot, entry.license_evidence!);
    if (existsSync(evidence)) {
      expect(sha256(evidence)).toBe(entry.license_evidence_sha256);
      expect(readFileSync(evidence, 'utf8')).toContain('creativecommons.org/licenses/by/4.0/');
    }
    expect(entry.used_in).toHaveLength(2);
    for (const owner of entry.used_in!)
      expect(readFileSync(resolve(repositoryRoot, owner), 'utf8')).toContain(entry.asset);
    return;
  }
  const nativeVlmFigure = {
    '00 Учебник/Assets/Figures/curated/vlm-training-2026-11-13/internvl35-training-process.png': {
      journal: '2026-11-13-internvl-training-figure-additions.json', anchor: 'S2.F3',
      image: '2508.18265v1/training_process.png', size: [1252, 317],
      sha: '5234140e21251cbfe871417a995edf37f8ee968b44586f2b49253c433ebb14db',
    },
    '00 Учебник/Assets/Figures/curated/vlm-training-2026-11-13/molmopoint-guisyn-figure3.png': {
      journal: '2026-11-13-pixmo-data-figure-additions.json', anchor: 'S4.F3',
      image: '2603.28069v1/MolmoPSyn_overview.png', size: [1536, 456],
      sha: 'dcb476acd4a19140b49e2ef0d9d69611e88833f402dd25f7789451d6ea9aad6d',
    },
    '00 Учебник/Assets/Figures/curated/vlm-eval-2026-11-13/mm-detect-contamination-figure1.png': {
      journal: '2026-11-13-vlm-evaluation-contamination-additions.json', anchor: 'S1.F1',
      image: '2411.03823v3/fig1_arr.png', size: [551, 354],
      sha: '26bb5ad1fbe11732b8b742269fe9d983d66dc5984d2a8c7da6ce23a1cae6ccf8',
    },
    '00 Учебник/Assets/Figures/curated/vlm-connectors-2026-11-13/pheye-dense-cross-attention.png': {
      journal: '2026-11-13-flamingo-molmo-figure-additions.json', anchor: 'S2.F2',
      image: '2501.02584v1/Images/cross-dense.png', size: [819, 480],
      sha: 'f74aaed3c0c517217b2449ad8ae4b2f89c2bff39981908a6310243e1a68e4bfc',
    },
  } as Record<string, {journal: string; anchor: string; image: string; size: number[]; sha: string}>;
  if (Object.hasOwn(nativeVlmFigure, entry.asset)) {
    const { journal, anchor, image, size, sha } = nativeVlmFigure[entry.asset]!;
    const review = JSON.parse(readFileSync(resolve(repositoryRoot,
      'docs/audit', journal), 'utf8')) as { figure: CuratedAsset };
    expect(entry).toEqual(review.figure);
    expect(entry).toMatchObject({
      license: 'CC BY 4.0', rights_status: 'source-license-verified',
      derivation: 'source-raster-copy', dimensions_px: size,
      source_image_url: 'https://arxiv.org/html/' + image,
      sha256: sha,
    });
    const file = resolve(repositoryRoot, entry.asset);
    expect(entry.source_asset).toBe(entry.asset);
    expect(sha256(file)).toBe(entry.sha256);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(validateRenderedPngBytes(readFileSync(file))).toEqual(size);
    const license = resolve(repositoryRoot, entry.license_evidence!);
    if (existsSync(license)) {
      expect(sha256(license)).toBe(entry.license_evidence_sha256);
      const page = readFileSync(license, 'utf8');
      expect(page).toContain('License: CC BY 4.0');
      expect(page).toContain('id="' + anchor + '"');
      expect(page).toContain(image);
    }
    expect(entry.used_in).toHaveLength(2);
    for (const owner of entry.used_in!) {
      const chapter = readFileSync(resolve(repositoryRoot, owner), 'utf8');
      expect(chapter).toContain(entry.asset);
      expect(chapter).toContain(entry.figure_url);
    }
    return;
  }
  const archivedVlmFigure = {
    '00 Учебник/Assets/Figures/curated/vlm-2026/internvl25-figure4-training-pipeline.png': {
      figure_url: 'https://arxiv.org/pdf/2412.05271v1#page=6',
      sha256: '5a869f582997e2f30df66ed7181a256ae1deb9d4e9fb89bdfa931a11dff2086c',
    },
    '00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure1-data-and-capabilities.png': {
      figure_url: 'https://arxiv.org/pdf/2409.17146v2#page=2',
      sha256: 'ff9247d31cc79ef048a409009c92894d1418f494814032aa4103871f13188b57',
    },
    '00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure11-openness.png': {
      figure_url: 'https://arxiv.org/pdf/2409.17146v2#page=19',
      sha256: 'a269b7cc10ee0eb9f7000baccda8d6bd9e513b5785fdf4c8b9af60bcaae5d209',
    },
    '00 Учебник/Assets/Figures/curated/vlm-2026/pix2struct-figure1-tasks.png': {
      figure_url: 'https://proceedings.mlr.press/v202/lee23g/lee23g.pdf#page=2',
      sha256: '888e2dc92f2dbba77850b8619ebbfafa3e2f5bb1608098c41a2d765f613edc52',
    },
    '00 Учебник/Assets/Figures/curated/vlm-2026/flamingo-figure3-architecture.png': {
      figure_url: 'https://arxiv.org/pdf/2204.14198v2#page=4',
      sha256: '87ac37b661368027927a867c8d4a173cd76e0ed988d6a1a9aebcb4b62c0c1d09',
    },
    '00 Учебник/Assets/Figures/curated/atlas-remainder-official/flamingo-architecture.png': {
      figure_url: 'https://arxiv.org/pdf/2204.14198v2#page=4',
      sha256: 'b0b7947d6d74f1441bd162e5d87307b4c32e2c5145774667f5f42c8df761914d',
    },
    '00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure5-image-tokenization.png': {
      figure_url: 'https://arxiv.org/pdf/2409.17146v2#page=9',
      sha256: '92d1f84d843ecd54ff46226df8ef6f424d285ed4c9a217882c5a8653df5c659e',
    },
  } as Record<string, {figure_url: string; sha256: string}>;
  if (Object.hasOwn(archivedVlmFigure, entry.asset)) {
    expect(entry).toMatchObject({
      publication_status: 'retired', rights_status: 'needs-confirmation', license: null,
      license_url: null, used_in: [], derivation: 'pdf-page-render-crop',
      ...archivedVlmFigure[entry.asset],
    });
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/vlm-connectors-2026-11-13/')) {
    const review = JSON.parse(readFileSync(resolve(repositoryRoot,
      'docs/audit/2026-11-13-internvl-imagebind-figures-additions.json'), 'utf8')) as {figures: CuratedAsset[]};
    const pinned = review.figures.find(figure => figure.asset === entry.asset);
    expect(pinned).toBeDefined();
    expect(entry).toEqual(pinned);
    const figurePath = resolve(repositoryRoot, entry.asset);
    expect(sha256(figurePath)).toBe(entry.sha256);
    expectDecodableImage(figurePath);
    const internvl = entry.asset.endsWith('internvl35-figure2c.png');
    if (internvl) {
      expect(entry.license).toBe('CC BY 4.0');
      expect(entry.crop_box_px).toEqual([1310, 0, 1739, 557]);
      expect(entry.derivation).toBe('source-raster-crop');
      expect(validateRenderedPngBytes(readFileSync(figurePath))).toEqual([429, 557]);
    } else {
      expect(entry.asset.endsWith('panda-act-figure2.jpg')).toBe(true);
      expect(entry.license).toBe('CC BY-NC-ND 4.0');
      expect(entry.derivation).toBe('source-image-copy');
      expect(entry.source_asset).toBe(entry.asset);
      expect(entry.source_asset_sha256).toBe(entry.sha256);
      expect(entry.crop_box_px).toBeUndefined();
    }
    for (const [file, hash] of [[entry.source_asset, entry.source_asset_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]]) {
      const path = resolve(repositoryRoot, file!);
      if (existsSync(path)) expect(sha256(path)).toBe(hash);
    }
    for (const owner of entry.used_in ?? []) {
      expect(readFileSync(resolve(repositoryRoot, owner), 'utf8')).toContain(entry.asset);
    }
    return;
  }
  if (entry.id === 'curated-prokudin-gorskii-rgb-927a674c0a18') {
    const pin = '927a674c0a18d2e5701f9e80d0cd968adbd9627cb4ff84d10e19aef45171a513';
    expect(entry).toMatchObject({
      asset: '00 Учебник/Assets/Figures/curated/harvard-credits-2026-11-11/prokudin-gorsky-rgb.jpg',
      author: 'Sergei Prokudin-Gorskii; composite by Wikimedia Commons contributor Mdd4696',
      source_url: 'https://commons.wikimedia.org/w/index.php?title=File:Alleia_Hamerops_composite.jpg&oldid=1130485549',
      source_image_url: 'https://upload.wikimedia.org/wikipedia/commons/f/f3/Alleia_Hamerops_composite.jpg',
      sha256: pin, source_asset_sha256: pin, source_asset: entry.asset,
      bytes: 156688, dimensions_px: [1000, 250], derivation: 'source-raster-copy',
      license: 'Public domain (PDM 1.0)', license_url: 'https://creativecommons.org/publicdomain/mark/1.0/',
      license_evidence: '05 Источники/Originals/harvard-credits-2026-11-11/rgb-license.html',
      license_evidence_sha256: 'd472ebe6dd28092f64fdd820e623e30db0630239700e78fb7d2d52b78bab5088',
      metadata_status: 'verified', rights_status: 'source-license-verified',
      provenance_confirmation: 'author-archive-and-commons-file-history',
      used_in: ['05 Источники/Courses/Harvard ML Systems/vol1/frameworks.md'],
    });
    const file = resolve(repositoryRoot, entry.asset);
    expect(sha256(file)).toBe(pin);
    expectDecodableImage(file);
    const licenseSnapshot = resolve(repositoryRoot, entry.license_evidence!);
    if (existsSync(licenseSnapshot)) {
      expect(sha256(licenseSnapshot)).toBe(entry.license_evidence_sha256);
      expect(readFileSync(licenseSnapshot, 'utf8')).toContain(entry.license_url);
    }
    return;
  }
  const vllmArchive = vllmSourceLibrary.retired.find(row => row.asset === entry.asset);
  if (vllmArchive) {
    expect(entry).toMatchObject(vllmArchive);
    expect(entry.publication_status).toBe('retired');
    expect(entry.used_in).toEqual([]);
    const target = resolve(repositoryRoot, entry.asset);
    const bytes = readFileSync(target);
    expect(bytes.length).toBe(vllmArchive.bytes);
    expect(sha256(target)).toBe(vllmArchive.sha256);
    expect(createHash('sha1').update('blob '+bytes.length+'\0').update(bytes).digest('hex'))
      .toBe(vllmArchive.git_blob);
    expectDecodableImage(target);
    const owner = readFileSync(resolve(repositoryRoot, '05 Источники/Implementation studies/Inside vLLM.md'), 'utf8');
    expect(owner).not.toContain(entry.asset);
    expect(owner).toContain(vllmArchive.figure_url!);
    return;
  }
  const curvature = optimizerCurvatureAssets.find(row => row.asset === entry.asset);
  if (curvature) {
    expect(entry).toMatchObject(curvature);
    const image = resolve(repositoryRoot, curvature.asset);
    expect(sha256(image)).toBe(curvature.sha256);
    expectDecodableImage(image);
    if (curvature === optimizerCurvatureReview.retired) {
      expect(entry.used_in).toEqual([]);
      for (const owner of optimizerCurvatureReview.retired.former_used_in) {
        const chapter = readFileSync(resolve(repositoryRoot, owner), 'utf8');
        expect(chapter).not.toContain(entry.asset);
        expect(chapter).toContain('https://cs231n.github.io/assets/nn3/opt1.gif');
      }
    } else {
      const source = resolve(repositoryRoot, curvature.source_asset!);
      expect(sha256(source)).toBe(curvature.source_asset_sha256);
      const evidence = resolve(repositoryRoot, curvature.license_evidence!);
      expect(sha256(evidence)).toBe(curvature.license_evidence_sha256);
      expect(readFileSync(evidence, 'utf8')).toContain('Creative Commons Attribution');
      if (curvature.derivation === 'source-svg-copy') {
        expect(sha256(source)).toBe(curvature.sha256);
      } else {
        expect(pngDimensions(image)).toEqual(curvature.dimensions_px);
      }
      for (const owner of curvature.used_in!) {
        const chapter = readFileSync(resolve(repositoryRoot, owner), 'utf8');
        expect(chapter).toContain(curvature.asset + '|');
        expect(chapter).toContain(curvature.source_image_url!);
      }
    }
    return;
  }
  const recall = sequenceRecallReview.figures.find(row => row.asset === entry.asset);
  const retiredRecall = sequenceRecallReview.retired.find(row => row.asset === entry.asset);
  if (recall || retiredRecall) {
    const spec = (recall ?? retiredRecall)!;
    expect({ ...entry, figure_url: entry.figure_url ?? entry.source_url }).toMatchObject(spec);
    const image = resolve(repositoryRoot, spec.asset);
    expect(sha256(image)).toBe(spec.sha256);
    expectDecodableImage(image);
    if (recall) {
      expect(pngDimensions(image)).toEqual(recall.dimensions_px);
      const evidence = resolve(repositoryRoot, recall.license_evidence!);
      expect(sha256(evidence)).toBe(recall.license_evidence_sha256);
      expect(readFileSync(evidence, 'utf8')).toContain('License: CC BY 4.0');
      if (recall.source_asset)
        expect(sha256(resolve(repositoryRoot, recall.source_asset))).toBe(recall.source_asset_sha256);
      for (const owner of recall.used_in!) {
        const text = readFileSync(resolve(repositoryRoot, owner), 'utf8');
        expect(text).toContain(recall.asset + '|');
        expect(text).toContain(recall.figure_url!);
      }
    } else {
      expect(entry.used_in).toEqual([]);
      for (const owner of retiredRecall!.former_used_in) {
        const text = readFileSync(resolve(repositoryRoot, owner), 'utf8');
        expect(text).not.toContain(entry.asset);
        expect(text).toContain(retiredRecall!.figure_url!);
      }
    }
    return;
  }
  const lmcache = [...lmcacheTransferReview.figures, ...lmcacheTransferReview.retired]
    .find(row => row.asset === entry.asset);
  if (lmcache) {
    expect(entry).toMatchObject(lmcache);
    const path = resolve(repositoryRoot, entry.asset);
    expect(sha256(path)).toBe(lmcache.sha256);
    expect(pngDimensions(path)).toEqual(lmcache.dimensions_px);
    expectDecodableImage(path);
    if ('replacements' in lmcache) {
      expect(entry.publication_status).toBe('retired');
      expect(entry.used_in).toEqual([]);
      expect(entry.rights_status).toBe('source-republication-unverified');
      for (const [i, owner] of lmcache.former_used_in!.entries()) {
        const text = readFileSync(resolve(repositoryRoot, owner), 'utf8');
        expect(text).not.toContain(entry.asset);
        expect(text).toContain(lmcache.replacements[i]);
        expect(text).toContain(entry.source_url!);
      }
    } else {
      expect(entry.license).toBe('CC BY 4.0');
      expect(entry.source_asset_sha256).toBe(entry.sha256);
      expect(entry.rights_status).toBe('source-license-verified');
      for (const owner of lmcache.used_in!) {
        const text = readFileSync(resolve(repositoryRoot, owner), 'utf8');
        expect(text).toContain(entry.asset + '|');
        expect(text).toContain(entry.figure_url!);
      }
    }
    return;
  }
  const inferenceFigure = [inferenceTimelineReview.figure, ...inferenceTimelineReview.historical]
    .find(row => row.asset === entry.asset);
  if (inferenceFigure) {
    const reuse = vllmSourceLibrary.figures.find(row => row.asset === entry.asset);
    expect(entry).toMatchObject({ ...inferenceFigure, ...(reuse && { used_in: reuse.used_in }) });
    const path = resolve(repositoryRoot, entry.asset);
    expect(sha256(path)).toBe(inferenceFigure.sha256);
    expectDecodableImage(path);
    for (const owner of inferenceFigure.used_in!)
      expect(readFileSync(resolve(repositoryRoot, owner), 'utf8')).toContain(entry.asset);
    if ('replacement' in inferenceFigure) {
      expect(entry.publication_status).toBe('source-library-only');
      expect(entry.rights_status).toBe('source-republication-unverified');
      for (const owner of inferenceFigure.former_used_in!) {
        const text = readFileSync(resolve(repositoryRoot, owner), 'utf8');
        expect(text).not.toContain(entry.asset);
        expect(text).toContain(inferenceFigure.replacement);
        expect(text).toContain(entry.source_url!);
      }
    } else {
      expect(entry.license).toBe('CC BY-NC-ND 4.0');
      expect(entry.rights_status).toBe('source-license-verified');
      expect(entry.source_asset_sha256).toBe(entry.sha256);
      expect(readFileSync(path, 'utf8')).toContain('viewBox="0 0 680 350"');
    }
    return;
  }
  const scheduling = [...schedulingSourceReview.figures, ...schedulingSourceReview.retired]
    .find(row => row.asset === entry.asset);
  if (scheduling) {
    expect(entry).toMatchObject(scheduling);
    const path = resolve(repositoryRoot, entry.asset);
    expect(sha256(path)).toBe(scheduling.sha256);
    expect(pngDimensions(path)).toEqual(scheduling.dimensions_px);
    if (entry.publication_status === 'retired') {
      expect(entry.used_in).toEqual([]);
      expect(entry.rights_status).toBe('source-republication-unverified');
      for (const note of entry.former_used_in!) {
        const body = readFileSync(resolve(repositoryRoot, note), 'utf8');
        expect(body).not.toContain(entry.asset);
        expect(body).toContain(entry.source_url);
      }
    } else {
      expect(entry.rights_status).toBe('source-license-verified');
      expectDecodableImage(path);
      if (entry.license_evidence) {
        const license = resolve(repositoryRoot, entry.license_evidence);
        if (existsSync(license)) {
          expect(sha256(license)).toBe(entry.license_evidence_sha256);
          expect(readFileSync(license, 'utf8')).toContain('license: cc-by-nc-sa-4.0');
        }
      }
      for (const note of entry.used_in!)
        expect(readFileSync(resolve(repositoryRoot, note), 'utf8')).toContain(entry.asset + '|');
    }
    return;
  }
  const reconciled = reconciledReaderFigures.find(row => row.asset === entry.asset);
  if (reconciled) {
    expect(entry).toMatchObject(reconciled);
    const target = resolve(repositoryRoot, entry.asset);
    expect(sha256(target)).toBe(reconciled.sha256);
    expectDecodableImage(target);
    for (const note of reconciled.used_in!)
      expect(readFileSync(resolve(repositoryRoot, note), 'utf8')).toContain(entry.asset);
    if (reconciled.public_license_copy)
      expect(sha256(resolve(repositoryRoot, reconciled.public_license_copy))).toBe(reconciled.license_evidence_sha256);
    return;
  }
  const retiredQuant = quantSourceReview.historical.find(row => row.asset === entry.asset);
  if (retiredQuant) {
    expect(entry).toMatchObject(retiredQuant);
    expect(entry.publication_status).toBe('retired');
    expect(entry.used_in).toEqual([]);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(retiredQuant.sha256);
    for (const owner of retiredQuant.former_used_in!) {
      const text = readFileSync(resolve(repositoryRoot, owner), 'utf8');
      expect(text).not.toContain(entry.asset);
      expect(text).toContain(retiredQuant.figure_url!);
      for (const replacement of retiredQuant.replacements) expect(text).toContain(replacement + '|');
    }
    return;
  }
  const quantFigure = quantSourceReview.figures.find(row => row.asset === entry.asset);
  if (quantFigure) {
    expect(entry).toMatchObject(quantFigure);
    const target = resolve(repositoryRoot, entry.asset);
    expect(sha256(target)).toBe(quantFigure.sha256);
    expect(pngDimensions(target)).toEqual(quantFigure.dimensions_px);
    expectDecodableImage(target);
    for (const [path, digest] of [
      [quantFigure.license_evidence, quantFigure.license_evidence_sha256],
      [quantFigure.source_code, quantFigure.source_code_sha256],
    ]) {
      if (!path) continue;
      const snapshot = resolve(repositoryRoot, path);
      if (existsSync(snapshot) || path === quantFigure.public_license_copy)
        expect(sha256(snapshot)).toBe(digest);
    }
    for (const owner of quantFigure.used_in!) {
      const text = readFileSync(resolve(repositoryRoot, owner), 'utf8');
      expect(text).toContain(quantFigure.asset + '|');
      expect(text).toContain(quantFigure.figure_url!);
      expect(text).toContain(quantFigure.public_license_copy?.replace('site/public', '') ?? quantFigure.license_url!);
    }
    return;
  }
  const archivedFlash = flashSourceReview.historical.find(row => row.asset === entry.asset);
  if (archivedFlash) {
    const formerOwners = archivedFlash.replacement
      ? [...archivedFlash.former_used_in!, '03 Исследовательские линии/Эффективный attention.md']
      : archivedFlash.former_used_in!;
    expect(entry).toMatchObject({ ...archivedFlash, former_used_in: formerOwners });
    expect(entry.publication_status).toBe('retired');
    expect(entry.used_in).toEqual([]);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(archivedFlash.sha256);
    for (const owner of formerOwners) {
      const body = readFileSync(resolve(repositoryRoot, owner), 'utf8');
      expect(body).not.toContain(entry.asset);
      if (archivedFlash.replacement) expect(body).toContain(archivedFlash.replacement);
      else expect(body).toContain(archivedFlash.figure_url!);
    }
    return;
  }
  const flash = flashSourceReview.figures.find(row => row.asset === entry.asset);
  const sourceFigure = flash ?? (entry.asset === gptRouteFigure.asset ? gptRouteFigure : undefined);
  if (sourceFigure) {
    const owners = sourceFigure.asset.endsWith('/fa1-tiling-loops.png')
      ? [...sourceFigure.used_in!, '03 Исследовательские линии/Эффективный attention.md']
      : sourceFigure.used_in!;
    expect(entry).toMatchObject({ ...sourceFigure, used_in: owners });
    const target = resolve(repositoryRoot, sourceFigure.asset);
    const source = resolve(repositoryRoot, sourceFigure.source_asset!);
    expect(sha256(target)).toBe(sourceFigure.sha256);
    expect(pngDimensions(target)).toEqual(sourceFigure.dimensions_px);
    expectDecodableImage(target);
    if (existsSync(source)) expect(sha256(source)).toBe(sourceFigure.source_asset_sha256);
    if (sourceFigure.license_evidence) {
      const evidence = resolve(repositoryRoot, sourceFigure.license_evidence);
      if (existsSync(evidence)) expect(sha256(evidence)).toBe(sourceFigure.license_evidence_sha256);
    }
    for (const owner of owners) {
      const body = readFileSync(resolve(repositoryRoot, owner), 'utf8');
      expect(body).toContain(sourceFigure.asset + '|');
      expect(body).toContain(sourceFigure === gptRouteFigure ? sourceFigure.source_image_url! : sourceFigure.figure_url!);
      expect(body).toContain(sourceFigure.public_license_copy?.replace('site/public', '') ?? sourceFigure.license_url!);
    }
    return;
  }
  const chen = chenSourceFigures.find(row => row.asset === entry.asset);
  if (chen) {
    expect(entry).toMatchObject(chen);
    const file = resolve(repositoryRoot, entry.asset);
    const bytes = readFileSync(file);
    expect(bytes.length).toBe(chen.bytes);
    expect(sha256(file)).toBe(chen.sha256);
    expect(createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex'))
      .toBe(chen.git_blob);
    expect(validateRenderedPngBytes(bytes)).toEqual(chen.dimensions_px);
    expect(readFileSync(resolve(repositoryRoot, chen.source_asset))).toEqual(bytes);
    const paper = resolve(repositoryRoot, chen.source_parent_asset);
    if (existsSync(paper)) expect(sha256(paper)).toBe(chen.source_parent_sha256);
    if (chen.publication_status === 'retired') {
      expect(entry.used_in).toEqual([]);
      expect(chen.replacement).toBeDefined();
      for (const owner of chenSourceFigures[1].used_in!) {
        const body = readFileSync(resolve(repositoryRoot, owner), 'utf8');
        expect(body).not.toContain(entry.asset);
        expect(body).toContain(chen.replacement!);
      }
    } else {
      expect(chen.used_in?.length).toBeGreaterThanOrEqual(2);
      for (const owner of chen.used_in!) {
        const body = readFileSync(resolve(repositoryRoot, owner), 'utf8');
        expect(body).toContain(entry.asset);
      }
    }
    return;
  }
  const nvidia = distributedNvidiaFigures.find(row => row.asset === entry.asset);
  if (nvidia) {
    expect(entry).toMatchObject(nvidia);
    const file = resolve(repositoryRoot, entry.asset);
    const bytes = readFileSync(file);
    expect(bytes.byteLength).toBe(nvidia.bytes);
    expect(sha256(file)).toBe(nvidia.sha256);
    expect(createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex'))
      .toBe(nvidia.git_blob);
    if (bytes[28] === 1) {
      // NCCL's originals use Adam7; the PDF-crop decoder only handles
      // non-interlaced PNGs. Check the pinned bytes and original PNG layout.
      expect(pngDimensions(file)).toEqual(nvidia.dimensions_px);
      expectDecodableImage(file);
    } else {
      expect(validateRenderedPngBytes(bytes)).toEqual(nvidia.dimensions_px);
    }
    expect(sha256(resolve(repositoryRoot, nvidia.license_evidence))).toBe(nvidia.license_evidence_sha256);
    const source = resolve(repositoryRoot, nvidia.source_asset);
    if (existsSync(source)) expect(readFileSync(source)).toEqual(bytes);
    expect(nvidia.used_in).toHaveLength(2);
    for (const owner of nvidia.used_in!) {
      const body = readFileSync(resolve(repositoryRoot, owner), 'utf8');
      expect(body).toContain(entry.asset + '|');
      expect(body).toContain(nvidia.license_evidence.replace('site/public', ''));
    }
    return;
  }
  const activeSequence = sequenceFigureReview.figures.find(row => row.asset === entry.asset);
  const retiredSequence = sequenceFigureReview.historical.find(row => row.asset === entry.asset);
  const sequence = activeSequence ?? retiredSequence;
  if (sequence) {
    expect(entry).toMatchObject(sequence);
    const file = resolve(repositoryRoot, entry.asset);
    expect(sha256(file)).toBe(sequence.sha256);
    expectDecodableImage(file);
    if (sequence.dimensions_px) expect(pngDimensions(file)).toEqual(sequence.dimensions_px);
    if (sequence.source_asset && sequence.source_asset_sha256) {
      const source = resolve(repositoryRoot, sequence.source_asset);
      if (existsSync(source)) expect(sha256(source)).toBe(sequence.source_asset_sha256);
    }
    if (retiredSequence) {
      expect(entry).toMatchObject({ publication_status: 'retired', used_in: [] });
      for (const former of retiredSequence.former_used_in) {
        const note = readFileSync(resolve(repositoryRoot, former), 'utf8');
        expect(note).not.toContain(entry.asset);
        expect(note).toContain(retiredSequence.replacement);
      }
    } else {
      expect(sequence.used_in?.length).toBeGreaterThan(0);
      for (const owner of sequence.used_in!)
        expect(readFileSync(resolve(repositoryRoot, owner), 'utf8')).toContain(entry.asset);
    }
    return;
  }
  const original = servingRepositoryOriginals.find(row => row.asset === entry.asset);
  if (original) {
    const bytes = readFileSync(resolve(repositoryRoot, entry.asset));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(original.sha256);
    expect(createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex'))
      .toBe(original.git_blob);
    expect(entry).toMatchObject({
      sha256: original.sha256, source_asset: original.source_asset,
      source_asset_sha256: original.source_asset_sha256, commit: original.git_commit,
      license: 'Apache-2.0', license_url: original.license_url,
      rights_status: 'source-license-verified', provenance_confirmation: 'source-license-verified',
    });
    const snapshot = resolve(repositoryRoot, original.source_asset);
    if (existsSync(snapshot)) expect(readFileSync(snapshot)).toEqual(bytes);
    expect(sha256(resolve(repositoryRoot, original.license_file))).toBe(original.license_sha256);
    if (original === servingRepositoryReview.archivedDynamoOverview) {
      expect(entry).toMatchObject({
        source_url: original.upstream_url, publication_status: 'retired', used_in: [],
      });
      expect(entry).toHaveProperty('replacement', servingRepositoryReview.figures[1].asset);
    } else {
      expect(entry).toMatchObject({
        source_image_url: original.upstream_url, git_blob: original.git_blob,
        license_evidence: original.license_file, license_evidence_sha256: original.license_sha256,
        metadata_status: 'verified',
      });
      expect(entry.used_in).toHaveLength(2);
      for (const owner of entry.used_in!) {
        const chapter = readFileSync(resolve(repositoryRoot, owner), 'utf8');
        expect(chapter).toContain(entry.asset + '|');
        expect(chapter).toContain(original.license_file.replace('site/public', ''));
      }
    }
    return;
  }
  if (entry.id === 'curated-pp-doclayout-v2-reading-order-f02cc9356b2a') {
    const version = 'https://arxiv.org/html/2510.14528v1';
    const digest = 'f02cc9356b2a659757ed83f1477881a0cd95b17e9408588daf0ade82811643e0';
    expect(entry).toMatchObject({
      asset: '00 Учебник/Assets/Figures/curated/paddleocr-vl-2026-11-11/pp-doclayout-v2-reading-order.png',
      author: 'Cheng Cui et al., PaddlePaddle Team, Baidu',
      source_url: version + '#S2.F3',
      source_image_url: version + '/images/PP-DocLayoutV2.png',
      license: 'CC BY 4.0',
      license_url: 'https://creativecommons.org/licenses/by/4.0/',
      license_evidence: version,
      provenance_confirmation: 'source-license-verified',
      rights_status: 'source-license-verified',
      metadata_status: 'verified',
      derivation: 'source-raster-copy',
      source_asset_sha256: digest,
      sha256: digest,
      dimensions_px: [1440, 436],
      bytes: 177984,
    });
    expect(entry.crop_box_px).toBeUndefined();
    const bytes = readFileSync(resolve(repositoryRoot, entry.asset));
    expect(bytes.length).toBe(177984);
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(digest);
    expect(validateRenderedPngBytes(bytes)).toEqual([1440, 436]);
    expect(entry.used_in).toHaveLength(2);
    for (const owner of entry.used_in!)
      expect(readFileSync(resolve(repositoryRoot, owner), 'utf8')).toContain(entry.asset);
    return;
  }
  const retiredVlm = retiredVlmFigures.find(row => row.asset === entry.asset);
  if (retiredVlm) {
    expect(entry).toMatchObject(retiredVlm);
    expect(entry).toMatchObject({
      publication_status: 'retired', rights_status: 'needs-confirmation', license: null, used_in: [],
    });
    const file = resolve(repositoryRoot, entry.asset);
    expect(sha256(file)).toBe(retiredVlm.sha256);
    expect(pngDimensions(file)).toEqual(retiredVlm.dimensions_px);
    expectDecodableImage(file);
    for (const former of retiredVlm.former_used_in ?? []) {
      const reader = readFileSync(resolve(repositoryRoot, former), 'utf8');
      expect(reader).not.toContain(entry.asset);
      expect(reader).toContain(retiredVlm.figure_url);
    }
    return;
  }
  const pdfEvidence = legacyPdfFigureEvidence.find(({ record }) => record.asset === entry.asset);
  if (pdfEvidence) {
    const expected = pdfEvidence.record;
    expect(entry).toMatchObject(expected);
    const file = resolve(repositoryRoot, entry.asset);
    expect(sha256(file)).toBe(expected.sha256);
    expect(pngDimensions(file)).toEqual(expected.dimensions_px);
    expectDecodableImage(file);
    if (expected.source_asset_sha256 && expected.source_asset) {
      const pdf = resolve(repositoryRoot, expected.source_asset);
      if (existsSync(pdf)) expect(sha256(pdf)).toBe(expected.source_asset_sha256);
    }
    for (const owner of expected.used_in ?? [])
      expect(readFileSync(resolve(repositoryRoot, owner), 'utf8')).toContain(entry.asset);
    return;
  }
  if (validateFinalSourceFigure(entry, repositoryRoot)) return;
  const alammarDirectory = '00 Учебник/Assets/Figures/curated/diffusion-alammar-2026-11-11/';
  if (entry.asset.startsWith(alammarDirectory)) {
    const name = entry.asset.slice(alammarDirectory.length);
    const original = stableDiffusionAlammarFigures[name];
    expect(original, name).toBeDefined();
    const article = 'https://jalammar.github.io/illustrated-stable-diffusion/';
    const evidence = 'publishing/fixtures/alammar-stable-diffusion-license.txt';
    expect(entry).toMatchObject({
      author: 'Jay Alammar',
      source_url: article + '#' + original.section,
      source_image_url: 'https://jalammar.github.io/images/stable-diffusion/' + name,
      source_blob_sha1: original.blob,
      source_asset_sha256: original.sha256, sha256: original.sha256,
      dimensions_px: original.dimensions,
      derivation: 'source-raster-copy', metadata_status: 'verified',
      rights_status: 'source-license-verified', provenance_confirmation: 'source-license-verified',
      license: 'CC BY-NC-SA 4.0', license_url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
      license_evidence: evidence, license_evidence_url: article,
      license_evidence_sha256: '728fee3c0dafb1193b31bbee494c93327cb8a1c90699f50c2d47942f4b7ddc7c',
    });
    expect(entry.crop_box_px).toBeUndefined();
    expect(sha256(resolve(repositoryRoot, evidence))).toBe(entry.license_evidence_sha256);
    const bytes = readFileSync(resolve(repositoryRoot, entry.asset));
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(original.sha256);
    expect(createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex')).toBe(original.blob);
    expect(validateRenderedPngBytes(bytes)).toEqual(original.dimensions);
    return;
  }
  const reviewed = reviewedLegacySources.find(row => row.asset === entry.asset);
  if (reviewed) {
    expect(entry).toMatchObject({
      source_image_url: reviewed.sourceImage, source_asset: reviewed.sourceAsset,
      source_asset_sha256: reviewed.sha256, sha256: reviewed.sha256, bytes: reviewed.bytes,
      license: reviewed.license, license_url: reviewed.licenseUrl, license_evidence: reviewed.licenseEvidence,
      metadata_status: 'verified', provenance_confirmation: 'source-license-verified',
      rights_status: 'source-license-verified', derivation: reviewed.viewBox ? 'source-svg-copy' : 'source-raster-copy',
    });
    expect(entry.crop_box_px).toBeUndefined();
    if (reviewed.commit) {
      expect(entry.commit).toBe(reviewed.commit);
      expect(entry.source_image_url).toContain('/' + reviewed.commit + '/');
    }
    const file = resolve(repositoryRoot, entry.asset);
    expect(sha256(file)).toBe(reviewed.sha256);
    expect(readFileSync(file).byteLength).toBe(reviewed.bytes);
    expect(sha256(resolve(repositoryRoot, reviewed.sourceAsset))).toBe(reviewed.sha256);
    expectDecodableImage(file);
    if (reviewed.dimensions) {
      expect(entry.dimensions_px).toEqual(reviewed.dimensions);
      if (file.endsWith('.png')) expect(pngDimensions(file)).toEqual(reviewed.dimensions);
    }
    if (reviewed.viewBox) {
      expect(entry.source_viewbox).toEqual(reviewed.viewBox);
      const viewBox = readFileSync(file, 'utf8').match(/<svg\b[^>]*\bviewBox=["']([^"']+)/)?.[1];
      expect(viewBox?.trim().split(/[\s,]+/).map(Number)).toEqual(reviewed.viewBox);
    }
    if (reviewed.licenseEvidenceSha256) {
      expect(entry.license_evidence_sha256).toBe(reviewed.licenseEvidenceSha256);
      if (!reviewed.licenseEvidence.startsWith('https://'))
        expect(sha256(resolve(repositoryRoot, reviewed.licenseEvidence))).toBe(reviewed.licenseEvidenceSha256);
    }
    if (reviewed.publicNotice) {
      expect(entry.public_license_copy).toBe(reviewed.publicNotice);
      expect(sha256(resolve(repositoryRoot, reviewed.publicNotice))).toBe(reviewed.publicNoticeSha256);
    }
    return;
  }
  const relativeAsset = entry.asset.replace(/^00 Учебник\/Assets\/Figures\/curated\//, '');
  const retired = retiredFigureRightsReview.find(([asset]) => asset === relativeAsset);
  if (retired) {
    expect(entry).toMatchObject({
      publication_status: 'retired', metadata_status: 'needs-confirmation',
      rights_status: retired[2], used_in: [], sha256: retired[1],
    });
    requireNonEmptyString(entry.retirement_reason, 'retirement_reason');
    expect(entry.source_url).toMatch(/^https:\/\//);
    const file = resolve(repositoryRoot, entry.asset);
    expect(sha256(file)).toBe(retired[1]);
    expectDecodableImage(file);
    return;
  }
  const d2l = d2lLegacyOriginals.find(([asset]) => asset === relativeAsset);
  if (d2l) {
    const license = '05 Источники/Originals/neural-language-model-2026-10-21/d2l-LICENSE.txt';
    expect(entry).toMatchObject({
      source_image_url: d2l[2].startsWith('/') ? 'https://d2l.ai' + d2l[2] : d2l[2],
      sha256: d2l[1], source_asset_sha256: d2l[1],
      derivation: 'source-svg-copy', metadata_status: 'verified',
      license: 'CC BY-SA 4.0', license_url: 'https://creativecommons.org/licenses/by-sa/4.0/',
      license_evidence: license, rights_status: 'source-license-verified',
      license_evidence_sha256: 'a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c',
    });
    expect(sha256(resolve(repositoryRoot, license))).toBe(entry.license_evidence_sha256);
    const file = resolve(repositoryRoot, entry.asset);
    expect(sha256(file)).toBe(d2l[1]);
    expectDecodableImage(file);
    return;
  }
  const chipHuyen = finalRetiredFigures[0];
  if (entry.asset === chipHuyen.asset) {
    expect(entry).toMatchObject({
      ...chipHuyen, author: 'Chip Huyen', license: null, license_url: null,
      source_url: 'https://huyenchip.com/2023/05/02/rlhf.html',
      source_image_url: 'https://huyenchip.com/assets/pics/rlhf/1-chatgpt-training.png',
      publication_status: 'retired', used_in: [], metadata_status: 'needs-confirmation',
      rights_status: 'needs-confirmation', provenance_confirmation: 'third-party-figure-origin-verified',
    });
    const file = resolve(repositoryRoot, entry.asset);
    expect(sha256(file)).toBe(chipHuyen.sha256);
    expectDecodableImage(file);
    return;
  }
  const agentPaperFigure = archivedAgentPaperFigures.find(row => row.asset === entry.asset);
  if (agentPaperFigure) {
    expect(entry).toMatchObject({
      source_url: agentPaperFigure.figure_url.split('#')[0].replace('/html/', '/abs/'),
      figure_url: agentPaperFigure.figure_url, sha256: agentPaperFigure.sha256,
      license: null, license_url: null,
      license_evidence_url: 'https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html',
      publication_status: 'retired', used_in: [],
      former_used_in: [`Papers/${agentPaperFigure.owner}.md`],
      metadata_status: 'needs-confirmation', rights_status: 'needs-confirmation',
      provenance_confirmation: 'third-party-figure-origin-verified',
    });
    requireNonEmptyString(entry.retirement_reason, 'retirement_reason');
    const file = resolve(repositoryRoot, entry.asset);
    expect(sha256(file)).toBe(agentPaperFigure.sha256);
    expectDecodableImage(file);
    return;
  }
  if (entry.asset.startsWith(foundationsFigurePath)) {
    validateFoundationsFigure(entry, repositoryRoot);
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/source-first-37-40-audit/longformer-attention-patterns.png') {
    expect(entry).toMatchObject({
      source_asset: 'raw/papers/external/2004.05150.pdf',
      source_url: 'https://arxiv.org/abs/2004.05150v2',
      figure_url: 'https://arxiv.org/html/2004.05150v2#S1.F2',
      license_url: null,
      license_evidence_url: 'https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html',
      derivation: 'pdf-page-render-crop', publication_status: 'retired', used_in: [],
      provenance_confirmation: 'third-party-figure-origin-verified',
      rights_status: 'needs-confirmation', metadata_status: 'needs-confirmation',
      sha256: 'f4ada49a4a88b0d324d4fba47ca41ad6acffc44f7aa4c5aac07b2fbd82d5b533',
    });
    requireNonEmptyString(entry.retirement_reason, 'retirement_reason');
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  const originalHeadFigure = multiheadOriginals.find(row => row.asset === entry.asset);
  if (originalHeadFigure) {
    expect(entry).toMatchObject(originalHeadFigure);
    const output = resolve(repositoryRoot, entry.asset);
    const source = resolve(repositoryRoot, originalHeadFigure.source_asset);
    const policy = resolve(repositoryRoot, originalHeadFigure.license_evidence);
    expect(sha256(output)).toBe(originalHeadFigure.sha256);
    expect(sha256(source)).toBe(originalHeadFigure.source_asset_sha256);
    expect(sha256(policy)).toBe(originalHeadFigure.license_evidence_sha256);
    expect(pngDimensions(output)).toEqual(originalHeadFigure.dimensions_px);
    if ('crop_box_px' in originalHeadFigure) {
      expect(readFileSync(policy,'utf8')).toContain('creativecommons.org/licenses/by/4.0/');
      const [x,y,right,bottom]=originalHeadFigure.crop_box_px;
      const expected=execFileSync(process.env.PDFTOPPM ?? 'pdftoppm',[
        '-f','3','-l','3','-r','360','-singlefile',
        '-x',String(x),'-y',String(y),'-W',String(right-x),'-H',String(bottom-y),'-png',source,
      ],{env:{...process.env,LC_ALL:'C',LANG:'C'},maxBuffer:1024*1024});
      expect(readFileSync(output)).toEqual(expected);
    } else {
      expect(readFileSync(policy,'utf8')).toContain('src="1706.03762v7/Figures/ModalNet-20.png" id="S3.F2.g2"');
      expect(readFileSync(policy,'utf8')).toContain('solely for use in journalistic or scholarly works');
      expect(readFileSync(output)).toEqual(readFileSync(source));
    }
    return;
  }
  if (entry.asset === "00 Учебник/Assets/Figures/curated/transformer-gpt-bridge-2026-11-10/vaswani-transformer-figure1.png") {
    const expected = {
      asset: "00 Учебник/Assets/Figures/curated/transformer-gpt-bridge-2026-11-10/vaswani-transformer-figure1.png",
      source_url: "https://arxiv.org/html/1706.03762v7#S3.F1",
      source_image_url: "https://arxiv.org/html/1706.03762v7/Figures/ModalNet-21.png",
      source_asset: "05 Источники/Originals/transformer-gpt-bridge-2026-11-10/ModalNet-21.png",
      source_asset_sha256: "56d69ccc192b90c3b93e5e285ff892f74753f970103939b40291945e106e9f79",
      sha256: "56d69ccc192b90c3b93e5e285ff892f74753f970103939b40291945e106e9f79",
      source_page_snapshot: "05 Источники/Originals/transformer-gpt-bridge-2026-11-10/vaswani-v7.html",
      source_page_sha256: "fe1288e9c6f8322156429ab44bf5ced203155f8ecc7ea9fe6cfd125872c72a3b",
      license: "Google permission for attributed scholarly or journalistic figure reuse",
      license_url: "https://arxiv.org/html/1706.03762v7#p1.1",
      license_evidence: "05 Источники/Originals/transformer-gpt-bridge-2026-11-10/vaswani-v7.html",
      license_evidence_sha256: "fe1288e9c6f8322156429ab44bf5ced203155f8ecc7ea9fe6cfd125872c72a3b",
      derivation: "source-png-copy",
      metadata_status: "verified",
      rights_status: "source-license-verified",
      used_in: ["00 Учебник/05 Attention и Transformer/03 Полный Transformer.md","en/00 Textbook/05 Attention and Transformer/05 The complete Transformer.md"],
    };
    expect(entry).toMatchObject(expected);
    const image = resolve(repositoryRoot, entry.asset);
    const original = resolve(repositoryRoot, expected.source_asset);
    const page = resolve(repositoryRoot, expected.source_page_snapshot);
    expect(sha256(image)).toBe(expected.sha256);
    expect(readFileSync(image)).toEqual(readFileSync(original));
    expect(sha256(page)).toBe(expected.source_page_sha256);
    expect(readFileSync(page, 'utf8')).toContain('src="1706.03762v7/Figures/ModalNet-21.png" id="S3.F1.g1"');
    expect(readFileSync(page, 'utf8')).toContain('solely for use in journalistic or scholarly works');
    expect(pngDimensions(image)).toEqual([1520,2239]);
    return;
  }
  const bridge = seq2seqAttentionBridge.find(row => row.asset === entry.asset);
  if (bridge) {
    const used_in = bridge.asset.endsWith('/d2l-seq2seq-state.svg')
      ? [...bridge.used_in, '00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры.md']
      : bridge.used_in;
    expect(entry).toMatchObject({ ...bridge, used_in });
    const original = resolve(repositoryRoot, bridge.source_asset);
    const output = resolve(repositoryRoot, entry.asset);
    const policy = resolve(repositoryRoot, bridge.license_evidence);
    expect(sha256(original)).toBe(bridge.source_asset_sha256);
    expect(sha256(output)).toBe(bridge.sha256);
    expect(sha256(policy)).toBe(bridge.license_evidence_sha256);
    if ('viewBox' in bridge) {
      const page = resolve(repositoryRoot, bridge.source_page_snapshot);
      const marker = bridge.source_image_url.split('/').at(-1)!;
      expect(sha256(page)).toBe(bridge.source_page_sha256);
      expect(readFileSync(page, 'utf8')).toContain('/' + marker + '"');
      expect(readFileSync(output, 'utf8')).toContain('viewBox="' + bridge.viewBox.join(' ') + '"');
      expect(readFileSync(output)).toEqual(readFileSync(original));
    } else {
      expect(readFileSync(policy, 'utf8')).toContain('creativecommons.org/licenses/by-nc-sa/3.0/');
      const [x, y, right, bottom] = bridge.crop_box_px;
      const expected = execFileSync(process.env.PDFTOPPM ?? 'pdftoppm', [
        '-f','7','-l','7','-r','360','-singlefile',
        '-x',String(x),'-y',String(y),'-W',String(right-x),'-H',String(bottom-y),'-png',original,
      ], { env: {...process.env, LC_ALL:'C', LANG:'C'}, maxBuffer:1024*1024 });
      expect(readFileSync(output)).toEqual(expected);
      expect(pngDimensions(output)).toEqual([2260,610]);
    }
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/lstm-gru-author-2026-11-10/cho-gru-figure2.png') {
    expect(entry).toMatchObject({
      source_url: 'https://aclanthology.org/D14-1179/',
      source_image_url: 'https://aclanthology.org/D14-1179.pdf#page=3',
      source_asset: '05 Источники/Originals/lstm-gru-2026-10-21/cho-2014.pdf',
      source_asset_sha256: 'c948d89ccd33da2a48421b45445f1c30f5fe677a829af9e5e1b6f9ff75a14a81',
      sha256: '394298380b6a5a1176228d7d019f9bc5c60868e2fed2ca839e9e039b7863dba9',
      derivation: 'pdf-page-render-crop', source_page: 3, source_printed_page: 1726,
      render_dpi: 360, crop_box_px: [1676,290,2486,725], dimensions_px: [810,435],
      license: 'CC BY-NC-SA 3.0', license_url: 'https://creativecommons.org/licenses/by-nc-sa/3.0/',
      license_evidence_url: 'https://aclanthology.org/faq/copyright/',
      license_evidence: '05 Источники/Originals/architecture-patterns-2026-10-21/acl-copyright.html',
      license_evidence_sha256: '88d644d2e21542ef1cc6c304c264ce7543d81e87c9599aff7e1c68e7a6df9558',
      metadata_status: 'verified', rights_status: 'source-license-verified',
    });
    const source = resolve(repositoryRoot, entry.source_asset!);
    const image = resolve(repositoryRoot, entry.asset);
    const policy = resolve(repositoryRoot, entry.license_evidence!);
    expect(sha256(source)).toBe(entry.source_asset_sha256);
    expect(sha256(image)).toBe(entry.sha256);
    expect(sha256(policy)).toBe(entry.license_evidence_sha256);
    expect(readFileSync(policy, 'utf8')).toContain('creativecommons.org/licenses/by-nc-sa/3.0/');
    const expected = execFileSync(process.env.PDFTOPPM ?? 'pdftoppm', [
      '-f','3','-l','3','-r','360','-singlefile','-x','1676','-y','290','-W','810','-H','435','-png',source,
    ], { env: {...process.env, LC_ALL:'C', LANG:'C'}, maxBuffer:1024*1024 });
    expect(readFileSync(image)).toEqual(expected);
    expect(pngDimensions(image)).toEqual([810,435]);
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/bpe-author-2026-11-10/zouhar-bpe-figure1.png') {
    expect(entry).toMatchObject({
      source_url: 'https://aclanthology.org/2023.findings-acl.38/',
      source_image_url: 'https://aclanthology.org/2023.findings-acl.38v1.pdf#page=3',
      source_asset: '05 Источники/Originals/bpe-author-2026-11-10/zouhar-acl-v1.pdf',
      source_asset_sha256: 'fe26d0d23fa838b083b16df808694d22402021bcf047cf606366578e270637b9',
      sha256: '6f5fee35e162629529c6856640a1ac6b0d646963820d40e198a6a95289a0f1fd',
      license: 'CC BY 4.0', license_url: 'https://creativecommons.org/licenses/by/4.0/',
      license_evidence_url: 'https://aclanthology.org/faq/copyright/',
      license_evidence: '05 Источники/Originals/architecture-patterns-2026-10-21/acl-copyright.html',
      license_evidence_sha256: '88d644d2e21542ef1cc6c304c264ce7543d81e87c9599aff7e1c68e7a6df9558',
      derivation: 'pdf-page-render-crop', source_page: 3, source_printed_page: 600,
      render_dpi: 360, crop_box_px: [1555,1756,2605,2286], dimensions_px: [1050,530],
      metadata_status: 'verified', rights_status: 'source-license-verified',
    });
    const source = resolve(repositoryRoot, entry.source_asset!);
    const image = resolve(repositoryRoot, entry.asset);
    const policy = resolve(repositoryRoot, entry.license_evidence!);
    expect(sha256(source)).toBe(entry.source_asset_sha256);
    expect(sha256(image)).toBe(entry.sha256);
    expect(sha256(policy)).toBe(entry.license_evidence_sha256);
    expect(readFileSync(policy, 'utf8')).toContain('https://creativecommons.org/licenses/by/4.0/');
    const expected = execFileSync(process.env.PDFTOPPM ?? 'pdftoppm', [
      '-f','3','-l','3','-r','360','-singlefile','-x','1555','-y','1756','-W','1050','-H','530','-png',source,
    ], { env: {...process.env, LC_ALL:'C', LANG:'C'}, maxBuffer:1024*1024 });
    expect(readFileSync(image)).toEqual(expected);
    expect(pngDimensions(image)).toEqual([1050,530]);
    return;
  }
  const embeddingFigures = '00 Учебник/Assets/Figures/curated/embedding-author-figures-2026-11-10/';
  if (entry.asset.startsWith(embeddingFigures)) {
    const packet = '05 Источники/Originals/embedding-author-figures-2026-11-10/';
    const member = entry.asset.slice(embeddingFigures.length);
    const digests: Record<string, string> = {
      'd2l-skip-gram.svg': '30ec5c85ee6e49d9f68673c45e3915f9642d13f02127622ecea024ed4f453b8a',
      'd2l-cbow.svg': 'eb0a0cacb7c7bfe38f573511ee5d426f4ee3b24e32b0ce3e8ca37b3d9d586811',
      'alammar-word2vec-lookup-embeddings.png': 'cef0f9dd8bc6bcc5a2bab1301e2337202ef88539c233111b7fd021d34c5d9759',
      'glove-table1-ratios.png': '934cb328deaced4f89024b53a92f2a5e138aeb518df69735b25c3c9d0e1c0783',
      'glove-figure1-weighting.png': '4f2c6d394cf411cee74f2420368fea2b3c2f34549de071b66bdd9f15e71af220',
    };
    expect(digests[member]).toBeDefined();
    const image = resolve(repositoryRoot, entry.asset);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.sha256).toBe(digests[member]);
    expect(sha256(image)).toBe(digests[member]);
    const checkedSource = resolve(repositoryRoot, entry.source_asset!);
    expect(sha256(checkedSource)).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    const evidence = readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8');
    if (member.startsWith('d2l-')) {
      const kind = member === 'd2l-cbow.svg' ? 'cbow' : 'skip-gram';
      const anchor = kind === 'cbow' ? 'the-continuous-bag-of-words-cbow-model' : 'the-skip-gram-model';
      expect(entry.commit).toBe(d2lCommit);
      expect(entry.source_url).toBe('https://en.d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html#' + anchor);
      expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/d2l-ai/d2l-en/' + d2lCommit + '/img/' + kind + '.svg');
      expect(entry.source_asset).toBe(packet + member);
      expect(entry.source_page_snapshot).toBe('05 Источники/Originals/word-embeddings-2026-10-21/d2l-word2vec.md');
      const sourcePage = resolve(repositoryRoot, entry.source_page_snapshot!);
      expect(sha256(sourcePage)).toBe(entry.source_page_sha256);
      expect(readFileSync(sourcePage, 'utf8')).toContain('../img/' + kind + '.svg');
      expect(entry.source_asset_sha256).toBe(entry.sha256);
      expect(entry.derivation).toBe('source-svg-copy');
      expect(entry.license_evidence).toBe(packet + 'd2l-license');
      expect(evidence).toContain('Creative Commons Attribution-ShareAlike 4.0 International License');
      expect(entry.license).toBe('CC BY-SA 4.0');
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-sa/4.0/');
      expect(readFileSync(image, 'utf8')).toContain('viewBox="0 0 178 117"');
    } else if (member.startsWith('alammar-')) {
      expect(entry.source_url).toBe('https://jalammar.github.io/illustrated-word2vec/#word2vec-training-process');
      expect(entry.source_image_url).toBe('https://jalammar.github.io/images/word2vec/word2vec-lookup-embeddings.png');
      expect(entry.source_asset).toBe(packet + member);
      expect(entry.source_page_snapshot).toBe(packet + 'illustrated-word2vec.html');
      expect(entry.source_page_snapshot).toBe(entry.license_evidence);
      expect(entry.source_page_sha256).toBe(entry.license_evidence_sha256);
      expect(evidence).toContain('/images/word2vec/word2vec-lookup-embeddings.png');
      expect(evidence).toContain('creativecommons.org/licenses/by-nc-sa/4.0/');
      expect(entry.source_asset_sha256).toBe(entry.sha256);
      expect(entry.derivation).toBe('source-raster-copy');
      expect(entry.license).toBe('CC BY-NC-SA 4.0');
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-nc-sa/4.0/');
      expect(pngDimensions(image)).toEqual([1431,628]);
    } else {
      const isTable = member === 'glove-table1-ratios.png';
      const page = isTable ? 3 : 4;
      const [x, y, right, bottom] = isTable ? [614,626,2348,1008] : [1565,300,2565,790];
      expect(entry.source_url).toBe('https://aclanthology.org/D14-1162/');
      expect(entry.source_image_url).toBe('https://aclanthology.org/D14-1162.pdf#page=' + page);
      expect(entry.source_asset).toBe(packet + 'glove-author.pdf');
      expect(entry.source_asset_sha256).toBe('6631a1719122a89b3941d851a55ecf38c55971c36b30ae9c0cec617124056adf');
      expect(entry.derivation).toBe('pdf-page-render-crop');
      expect(entry.source_page).toBe(page);
      expect(entry.source_printed_page).toBe(1531 + page);
      expect(entry.crop_box_px).toEqual([x,y,right,bottom]);
      expect(entry.render_dpi).toBe(360);
      expect(entry.render_page_size_px).toEqual([2977,4210]);
      expect(entry.license).toBe('CC BY-NC-SA 3.0');
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-nc-sa/3.0/');
      expect(entry.license_evidence_url).toBe('https://aclanthology.org/faq/copyright/');
      expect(evidence).toContain('creativecommons.org/licenses/by-nc-sa/3.0/');
      const crop = execFileSync(process.env.PDFTOPPM ?? 'pdftoppm', [
        '-f', String(page), '-l', String(page), '-r', '360', '-singlefile',
        '-x', String(x), '-y', String(y), '-W', String(right - x), '-H', String(bottom - y),
        '-png', checkedSource,
      ], {env: {...process.env, LC_ALL: 'C', LANG: 'C'}, maxBuffer: 1024 * 1024});
      expect(readFileSync(image)).toEqual(crop);
      expect(pngDimensions(image)).toEqual([right-x,bottom-y]);
    }
    expectDecodableImage(image);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/deepseek-v2-mla-2026-11-09/')) {
    const specs = JSON.parse(readFileSync(resolve(repositoryRoot, 'publishing/tools/deepseek-v2-mla-originals.json'), 'utf8')) as CuratedAsset[];
    const spec = specs.find(row => row.asset === entry.asset);
    expect(spec).toBeDefined();
    expect(entry).toMatchObject(spec!);
    const path = resolve(repositoryRoot, entry.asset);
    expect(sha256(path)).toBe(spec!.sha256);
    expect(pngDimensions(path)).toEqual(spec!.dimensions_px);
    expect(sha256(resolve(repositoryRoot, spec!.license_evidence!))).toBe(spec!.license_evidence_sha256);
    expectDecodableImage(path);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/deepseek-originals-2026-11-09/')) {
    const sources = [
      {
        "id": "curated-deepseek-v3-benchmarks-d2d6b339d6eb",
        "asset": "00 Учебник/Assets/Figures/curated/deepseek-originals-2026-11-09/v3-figure1-benchmarks.png",
        "source_url": "https://github.com/deepseek-ai/DeepSeek-V3/blob/d5c08b384b4d4096e494d61ba7d329f593116872/DeepSeek_V3.pdf",
        "source_image_url": "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/d5c08b384b4d4096e494d61ba7d329f593116872/DeepSeek_V3.pdf#page=1",
        "sha256": "d2d6b339d6ebaafdaa8f9ff381c4128ca6e82654f55136f6de4028135569f1c6",
        "dimensions_px": [
          2285,
          1340
        ],
        "source_page": 1,
        "crop_box_px": [
          345,
          2310,
          2630,
          3650
        ],
        "license": "MIT",
        "derivation": "pdf-page-render-crop",
        "source_asset_sha256": "1392807ef6297deefc421cea6633af0b819e15a0b7a4a6b0adba4bccdf0ab273",
        "license_evidence": "site/public/licenses/deepseek-v3-d5c08b3-MIT.txt"
      },
      {
        "id": "curated-deepseek-v3-block-ad55ce4f20f2",
        "asset": "00 Учебник/Assets/Figures/curated/deepseek-originals-2026-11-09/v3-figure2-block.png",
        "source_url": "https://github.com/deepseek-ai/DeepSeek-V3/blob/d5c08b384b4d4096e494d61ba7d329f593116872/DeepSeek_V3.pdf",
        "source_image_url": "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/d5c08b384b4d4096e494d61ba7d329f593116872/DeepSeek_V3.pdf#page=7",
        "sha256": "ad55ce4f20f25e0f04fa439b3fb733557d254062df7d9f97d406f52915a4e809",
        "dimensions_px": [
          2285,
          1845
        ],
        "source_page": 7,
        "crop_box_px": [
          345,
          405,
          2630,
          2250
        ],
        "license": "MIT",
        "derivation": "pdf-page-render-crop",
        "source_asset_sha256": "1392807ef6297deefc421cea6633af0b819e15a0b7a4a6b0adba4bccdf0ab273",
        "license_evidence": "site/public/licenses/deepseek-v3-d5c08b3-MIT.txt"
      },
      {
        "id": "curated-deepseek-r1-zero-nature-f79140c320e1",
        "asset": "00 Учебник/Assets/Figures/curated/deepseek-originals-2026-11-09/r1-zero-nature-figure1.png",
        "source_url": "https://www.nature.com/articles/s41586-025-09422-z/figures/1",
        "source_image_url": "https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_Fig1_HTML.png",
        "sha256": "f79140c320e139280b30a7aeda5e886226ebaa06227742cf374507b03b8d0bcb",
        "dimensions_px": [
          1799,
          782
        ],
        "license": "CC BY 4.0",
        "derivation": "source-raster-copy",
        "source_asset_sha256": "f79140c320e139280b30a7aeda5e886226ebaa06227742cf374507b03b8d0bcb",
        "license_evidence": "https://www.nature.com/articles/s41586-025-09422-z#rightslink"
      },
      {
        "id": "curated-deepseek-v2-author-architecture-69910543d224",
        "asset": "00 Учебник/Assets/Figures/curated/deepseek-originals-2026-11-09/v2-author-architecture.png",
        "source_url": "https://github.com/deepseek-ai/DeepSeek-V2/blob/ec98ee3cbffc32104cd55dba8af884b3d772602a/figures/architecture.png",
        "source_image_url": "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V2/ec98ee3cbffc32104cd55dba8af884b3d772602a/figures/architecture.png",
        "sha256": "69910543d2248eb7d201a3aa164278fc24f886f86e5acd0273dc70fa5e9a728c",
        "dimensions_px": [
          1139,
          918
        ],
        "license": "MIT",
        "derivation": "source-raster-copy",
        "source_asset_sha256": "69910543d2248eb7d201a3aa164278fc24f886f86e5acd0273dc70fa5e9a728c",
        "license_evidence": "site/public/licenses/deepseek-v2-ec98ee3c-MIT.txt"
      }
    ];
    const source = sources.find(candidate => candidate.id === entry.id);
    expect(source).toBeDefined();
    expect(entry).toMatchObject({ ...source, metadata_status: 'verified', rights_status: 'source-license-verified' });
    const path = resolve(repositoryRoot, entry.asset);
    expect(sha256(path)).toBe(source!.sha256);
    const png = readFileSync(path);
    expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual(source!.dimensions_px);
    if (source!.license === 'MIT') {
      expect(sha256(resolve(repositoryRoot, source!.license_evidence))).toBe('6e4c38e1172f42fdbff13edf9a7a017679fb82b0fde415a3e8b3c31c6ed4a4e4');
    }
    expectDecodableImage(path);
    return;
  }
  if (/^curated-enigma-icml-figure[13]-/.test(entry.id)) {
    const crops = {
      'curated-enigma-icml-figure1-7d8da45c3b89': { page: 2, box: [208, 250, 2184, 702], name: 'enigma-figure1.png', hash: '7d8da45c3b8931b52c69df0b278ffae800e788b154083f997e6f258a52066d5f' },
      'curated-enigma-icml-figure3-09dee15a783d': { page: 5, box: [208, 250, 2232, 1092], name: 'enigma-figure3.png', hash: '09dee15a783dbd937c8fd325bdb68a03a940d0beb6afad4ea7214848aa880ad0' },
    };
    const crop = crops[entry.id as keyof typeof crops];
    expect(crop).toBeDefined();
    const asset = '00 Учебник/Assets/Figures/curated/computer-agents-2026-11-09/' + crop.name;
    const source = '05 Источники/Originals/enigma-2026-11-09/abramovich25a.pdf';
    const sourceHash = 'ef4914f8452197ab3123d4d014a5c7e618476cfca552f628b613f8fdb71d5b89';
    expect(entry).toMatchObject({
      asset, source_asset: source, source_asset_sha256: sourceHash, source_page: crop.page,
      author: 'Talor Abramovich et al.',
      source_url: 'https://proceedings.mlr.press/v267/abramovich25a.html',
      source_image_url: 'https://raw.githubusercontent.com/mlresearch/v267/7914d3407046b308db90001490534e422dab7d87/assets/abramovich25a/abramovich25a.pdf#page=' + crop.page,
      license: 'CC BY 4.0', license_url: 'https://creativecommons.org/licenses/by/4.0/',
      license_evidence: 'https://proceedings.mlr.press/pmlr-license-agreement.html',
      rights_evidence: 'https://icml.cc/Conferences/2025/AuthorInstructions',
      metadata_status: 'verified', provenance_confirmation: 'source-license-verified',
      derivation: 'pdf-page-render-crop', render_dpi: 288,
      crop_box_px: crop.box, render_page_size_px: [2448, 3168],
      sha256: crop.hash, used_in: ['00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents.md'],
    });
    const path = resolve(repositoryRoot, asset);
    expect(sha256(path)).toBe(crop.hash);
    expect(pngDimensions(path)).toEqual([crop.box[2] - crop.box[0], crop.box[3] - crop.box[1]]);
    expectDecodableImage(path);
    const archived = resolve(repositoryRoot, source);
    if (existsSync(archived)) expect(sha256(archived)).toBe(sourceHash);
    return;
  }
  if (/^curated-eagle3-(inference-figure5|mask-figure6)-/.test(entry.id)) {
    const originals = [{"id":"curated-eagle3-inference-figure5-1fc5aea1023b","asset":"00 Учебник/Assets/Figures/curated/eagle3-2026-11-09/eagle3-inference-figure5.png","title":"EAGLE-3 inference pipeline (Figure 5)","source_url":"https://arxiv.org/html/2503.01840v3#S3.F5","source_image_url":"https://arxiv.org/html/2503.01840v3/method.png","source_asset":"00 Учебник/Assets/Figures/curated/eagle3-2026-11-09/eagle3-inference-figure5.png","source_asset_sha256":"1fc5aea1023b11258bcc47d75f528b4e0224364089c2bbdba20a0c0aeb0f86c1","sha256":"1fc5aea1023b11258bcc47d75f528b4e0224364089c2bbdba20a0c0aeb0f86c1","derivation":"source-png-copy","dimensions_px":[647,987],"author":"Yuhui Li, Fangyun Wei, Chao Zhang and Hongyang Zhang","license":"CC BY 4.0","license_url":"https://creativecommons.org/licenses/by/4.0/","license_evidence":"05 Источники/Originals/eagle3-2026-11-09/openreview-4exx1hUffq.json","license_evidence_sha256":"44bd391f91ae1303eeeea81ad0e961fd97cf927ef57353bf1ab14554859d6064","rights_evidence":"https://openreview.net/forum?id=4exx1hUffq","rights_status":"source-license-verified","metadata_status":"verified","provenance_confirmation":"author-camera-ready-license","last_verified":"2026-11-09","modifications":"Unmodified original figure file from arXiv HTML v3; the authors publish the same figure in the CC BY 4.0 camera-ready article on OpenReview.","used_in":["00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md","en/00 Textbook/14 Inference and optimization/58 Speculative decoding.md"]},{"id":"curated-eagle3-mask-figure6-91c835f7a994","asset":"00 Учебник/Assets/Figures/curated/eagle3-2026-11-09/eagle3-training-mask-figure6.svg","title":"EAGLE-3 attention masks during training-time test (Figure 6)","source_url":"https://arxiv.org/html/2503.01840v3#S3.F6","source_image_url":"https://arxiv.org/html/2503.01840v3/train.svg","source_asset":"00 Учебник/Assets/Figures/curated/eagle3-2026-11-09/eagle3-training-mask-figure6.svg","source_asset_sha256":"91c835f7a9949fdd466d453204670652cab8174c088326f03d46ad040c3ee656","sha256":"91c835f7a9949fdd466d453204670652cab8174c088326f03d46ad040c3ee656","derivation":"source-svg-copy","viewBox":[0,0,919.5071,434.72798],"author":"Yuhui Li, Fangyun Wei, Chao Zhang and Hongyang Zhang","license":"CC BY 4.0","license_url":"https://creativecommons.org/licenses/by/4.0/","license_evidence":"05 Источники/Originals/eagle3-2026-11-09/openreview-4exx1hUffq.json","license_evidence_sha256":"44bd391f91ae1303eeeea81ad0e961fd97cf927ef57353bf1ab14554859d6064","rights_evidence":"https://openreview.net/forum?id=4exx1hUffq","rights_status":"source-license-verified","metadata_status":"verified","provenance_confirmation":"author-camera-ready-license","last_verified":"2026-11-09","modifications":"Unmodified original figure file from arXiv HTML v3; the authors publish the same figure in the CC BY 4.0 camera-ready article on OpenReview.","used_in":["00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md","en/00 Textbook/14 Inference and optimization/58 Speculative decoding.md"]}];
    const spec = originals.find(row => row.id === entry.id);
    expect(spec).toBeDefined();
    expect(entry).toMatchObject(spec!);
    const asset = resolve(repositoryRoot, spec!.asset);
    expect(sha256(asset)).toBe(spec!.sha256);
    expectDecodableImage(asset);
    const evidence = resolve(repositoryRoot, spec!.license_evidence);
    if (existsSync(evidence)) {
      expect(sha256(evidence)).toBe(spec!.license_evidence_sha256);
      expect(JSON.parse(readFileSync(evidence, 'utf8'))).toMatchObject({
        id: '4exx1hUffq', license: 'CC BY 4.0', domain: 'NeurIPS.cc/2025/Conference',
        signatures: ['NeurIPS.cc/2025/Conference/Submission13079/Authors'],
        content: { pdf: { value: '/pdf/0fe7af5ec1ec69cb936b30db6ec01cdec6fc5d81.pdf' } },
      });
    }
    return;
  }
  if (entry.id === 'curated-deepseek-v3-mtp-figure3-f39402216f59') {
    const commit = 'd5c08b384b4d4096e494d61ba7d329f593116872';
    const upstream = 'https://github.com/deepseek-ai/DeepSeek-V3/blob/' + commit;
    const raw = 'https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/' + commit;
    const asset = '00 Учебник/Assets/Figures/curated/ntp-2026-10-23/deepseek-v3-mtp-figure3.png';
    const notice = 'site/public/licenses/deepseek-v3-d5c08b3-MIT.txt';
    const digest = 'f39402216f591abb5928bbe867be30c7c3f9a6ffcfbc88ba6b34a228e3317903';
    expect(entry).toMatchObject({
      asset, commit, source_url: upstream + '/DeepSeek_V3.pdf',
      source_image_url: raw + '/DeepSeek_V3.pdf#page=10',
      source_asset_sha256: '1392807ef6297deefc421cea6633af0b819e15a0b7a4a6b0adba4bccdf0ab273',
      source_page: 10, render_dpi: 360, crop_box_px: [340, 400, 2640, 1445],
      dimensions_px: [2300, 1045], license: 'MIT', license_url: upstream + '/LICENSE-CODE',
      license_evidence: notice, public_license_copy: notice, sha256: digest,
      rights_status: 'source-license-verified', derivation: 'pdf-page-render-crop',
      used_in: ['00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md'],
    });
    const path = resolve(repositoryRoot, asset);
    expect(sha256(path)).toBe(digest);
    const png = readFileSync(path);
    expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual([2300, 1045]);
    expect(sha256(resolve(repositoryRoot, notice))).toBe('6e4c38e1172f42fdbff13edf9a7a017679fb82b0fde415a3e8b3c31c6ed4a4e4');
    expectDecodableImage(path);
    return;
  }
  if (entry.id === 'curated-eagle3-speedup-5e404ac75809') {
    const expected = {"id":"curated-eagle3-speedup-5e404ac75809","asset":"00 Учебник/Assets/Figures/curated/eagle3-2026-11-09/eagle3r.jpg","title":"EAGLE-3 greedy speedup across four target models","author":"Yuhui Li, Fangyun Wei, Chao Zhang and Hongyang Zhang","source_url":"https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/figs/eagle3r.jpg","source_image_url":"https://raw.githubusercontent.com/SafeAILab/EAGLE/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/figs/eagle3r.jpg","commit":"cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b","license":"Apache-2.0","license_url":"https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/LICENSE","license_copy":"site/public/licenses/eagle-cb7e0841-Apache-2.0.txt","license_copy_sha256":"2c042313b4fb4350fa09f81aabf6b77686855e1071a8af07845684e9cba6d3c7","source_page_snapshot":"05 Источники/Originals/eagle3-2026-11-09/README.md","source_page_sha256":"a59962f5c3b938b73252953119557cbeebfd4f13a1553e5f2fb4fe9358578a88","sha256":"5e404ac75809d8125e1c360c36054c2f90ebd72453a46a352609e538d91f6ba1","dimensions_px":[5862,1535],"derivation":"source-raster-copy","metadata_status":"verified","provenance_confirmation":"pinned-upstream-repository","last_verified":"2026-11-09","modifications":"Unmodified complete JPEG copied from the authors' official repository; it is not a crop of the arXiv Figure 2.","used_in":["00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md","en/00 Textbook/14 Inference and optimization/58 Speculative decoding.md"]};
    expect(entry).toMatchObject(expected);
    expect(entry.crop_box_px).toBeUndefined();
    const file = resolve(repositoryRoot, expected.asset);
    expect(sha256(file)).toBe(expected.sha256);
    expect(sha256(resolve(repositoryRoot, expected.license_copy))).toBe(expected.license_copy_sha256);
    const readme = resolve(repositoryRoot, expected.source_page_snapshot);
    if (existsSync(readme)) {
      expect(sha256(readme)).toBe(expected.source_page_sha256);
      expect(readFileSync(readme, 'utf8')).toContain('./figs/eagle3r.jpg');
    }
    expectDecodableImage(file);
    return;
  }
  if (entry.asset.startsWith(isoflopOriginalDir + '/')) {
    const original = isoflopOriginals.find(([name]) => entry.asset === isoflopOriginalDir + '/' + name);
    expect(original).toBeDefined();
    const [name, digest, figure, source] = original!;
    const owners = ['00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md'];
    if (name === 'training_flop_vs_model_size.png')
      owners.push('00 Учебник/_index.md', 'en/00 Textbook/_index.md');
    expect(entry).toMatchObject({
      source_asset: entry.asset, source_asset_sha256: digest, sha256: digest,
      source_url: figure, source_image_url: source, rights_evidence: figure.split('#')[0],
      license: 'CC BY 4.0', license_url: 'https://creativecommons.org/licenses/by/4.0/',
      metadata_status: 'verified', provenance_confirmation: 'source-license-verified',
      rights_status: 'source-license-verified', used_in: owners,
      derivation: name.endsWith('.svg') ? 'source-svg-copy' : 'source-png-copy',
    });
    expect(entry.crop_box_px).toBeUndefined();
    const target = resolve(repositoryRoot, entry.asset);
    expect(sha256(target)).toBe(digest);
    expectDecodableImage(target);
    return;
  }
  if (entry.id === 'curated-arithmetic-sparse-frontier-e309c11191da') {
    const original = {"asset":"00 Учебник/Assets/Figures/curated/arithmetic-attention-share-2026-11-09/prefilling_seq_length_impact.svg","author":"Piotr Nawrot, Robert Li, Renjie Huang, Sebastian Ruder, Kelly Marchisio and Edoardo M. Ponti","source_url":"https://arxiv.org/html/2504.17768v2#A2.F12","source_image_url":"https://arxiv.org/html/2504.17768v2/prefilling_seq_length_impact.svg","source_asset":"00 Учебник/Assets/Figures/curated/arithmetic-attention-share-2026-11-09/prefilling_seq_length_impact.svg","source_asset_sha256":"e309c11191da88331f9302fdb3983f75c68e6c8afe119b1721f98c3d11cfeef6","sha256":"e309c11191da88331f9302fdb3983f75c68e6c8afe119b1721f98c3d11cfeef6","license":"CC BY 4.0","license_url":"https://creativecommons.org/licenses/by/4.0/","rights_evidence":"https://arxiv.org/html/2504.17768v2","derivation":"source-svg-copy","metadata_status":"verified","provenance_confirmation":"source-license-verified","rights_status":"source-license-verified","used_in":["00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md"]};
    expect(entry).toMatchObject(original);
    const path = resolve(repositoryRoot, original.asset);
    expect(sha256(path)).toBe(original.sha256);
    expect(entry.crop_box_px).toBeUndefined();
    expectDecodableImage(path);
    return;
  }
  const scalingFigure = originalScalingMetricFigures.find(spec => spec.id === entry.id);
  if (scalingFigure) {
    expect(entry).toMatchObject({
      ...scalingFigure, source_asset_sha256: scalingFigure.sha256,
      rights_status: 'source-license-verified', provenance_confirmation: 'source-license-verified',
      metadata_status: 'verified', used_in: ['00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md'],
    });
    const asset = resolve(repositoryRoot, scalingFigure.asset);
    expect(sha256(asset)).toBe(scalingFigure.sha256);
    expect(sha256(resolve(repositoryRoot, scalingFigure.source_asset))).toBe(scalingFigure.sha256);
    expect(entry.crop_box_px).toBeUndefined();
    if (scalingFigure.dimensions_px) expect(pngDimensions(asset)).toEqual(scalingFigure.dimensions_px);
    expectDecodableImage(asset);
    return;
  }
  if (entry.id === 'curated-google-softmax-multiclass-7b0f01cc61b9') {
    const asset = '00 Учебник/Assets/Figures/curated/foundations-source-first/google-softmax-multiclass.png';
    const digest = '7b0f01cc61b917b9c76a698af949c21acce117dad1d46247f1fe028555f3ad2c';
    expect(entry).toMatchObject({
      asset, source_asset: asset, author: 'Google', sha256: digest, source_asset_sha256: digest,
      dimensions_px: [1850, 834],
      source_url: 'https://developers.google.com/machine-learning/crash-course/neural-networks/multi-class',
      source_image_url: 'https://developers.google.com/static/machine-learning/crash-course/neural-networks/images/one_vs_one_neural_net.png',
      rights_evidence: 'https://developers.google.com/machine-learning/crash-course/neural-networks/multi-class#one-versus-one-softmax',
      license: 'CC BY 4.0', license_url: 'https://creativecommons.org/licenses/by/4.0/',
      rights_status: 'source-license-verified', provenance_confirmation: 'source-license-verified',
      derivation: 'source-raster-copy', metadata_status: 'verified', used_in: ['00 Учебник/00 Математические и ML-основания/04 Функции потерь.md'],
    });
    expect(sha256(resolve(repositoryRoot, asset))).toBe(digest);
    expect(pngDimensions(resolve(repositoryRoot, asset))).toEqual([1850, 834]);
    expect(entry.crop_box_px).toBeUndefined();
    expectDecodableImage(resolve(repositoryRoot, asset));
    return;
  }
  if (entry.id === 'curated-cs229-bias-variance-0000000000ac') {
    const asset = '00 Учебник/Assets/Figures/curated/foundations-source-first/cs229-bias-variance-decomposition.png';
    expect(entry).toMatchObject({
      asset, author: 'Anand Avati, Stanford CS229',
      source_url: 'https://cs229.stanford.edu/summer2023/BiasVarianceAnalysis.pdf',
      source_page: 7, derivation: 'pdf-page-render-crop', publication_status: 'retired', used_in: [],
      former_used_in: ['00 Учебник/00 Математические и ML-основания/06 Переобучение, регуляризация и оценивание.md'],
      license: 'Unverified for redistribution', license_url: null,
      provenance_confirmation: 'source-linked-rights-unverified',
      rights_status: 'course-note-republication-unverified',
      sha256: '244d71e359d9a099496b02fa305b4fccffdaf968281dd411274f2782c3b7d493',
    });
    expect(sha256(resolve(repositoryRoot, asset))).toBe(entry.sha256);
    expectDecodableImage(resolve(repositoryRoot, asset));
    return;
  }
  if (entry.id === 'curated-sklearn-tree-bagging-8fa36b8ac57a') {
    const asset = '00 Учебник/Assets/Figures/curated/foundations-source-first/sklearn-tree-bagging-bias-variance.png';
    const commit = '199d116ab37beed506c63fc9f6a74f56e228de43';
    const digest = '8fa36b8ac57a9fdfa12f7856d4d55fb70f82dc09d62f8b1a70604473e06a3a08';
    const license = 'site/public/licenses/scikit-learn-1.9.1-BSD-3-Clause.txt';
    expect(entry).toMatchObject({
      asset, source_asset: asset, commit,
      author: 'The scikit-learn developers',
      source_url: 'https://scikit-learn.org/1.9/auto_examples/ensemble/plot_bias_variance.html',
      source_image_url: 'https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/' +
        commit + '/1.9/_images/sphx_glr_plot_bias_variance_001.png',
      sha256: digest, source_asset_sha256: digest, dimensions_px: [1000, 800],
      license: 'BSD-3-Clause',
      license_url: 'https://github.com/scikit-learn/scikit-learn/blob/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/COPYING',
      license_evidence: license, public_license_copy: license,
      license_evidence_sha256: '50d6a9d340f19ab355609917993114daf5f47e3161067bcf34955bbd05cd9cb0',
      provenance_confirmation: 'source-license-verified', rights_status: 'source-license-verified',
      metadata_status: 'verified', derivation: 'source-raster-copy',
      used_in: ['00 Учебник/00 Математические и ML-основания/06 Переобучение, регуляризация и оценивание.md'],
    });
    expect(sha256(resolve(repositoryRoot, asset))).toBe(digest);
    expect(sha256(resolve(repositoryRoot, license))).toBe(entry.license_evidence_sha256);
    expect(pngDimensions(resolve(repositoryRoot, asset))).toEqual([1000, 800]);
    expect(entry.crop_box_px).toBeUndefined();
    expectDecodableImage(resolve(repositoryRoot, asset));
    return;
  }
  if (entry.id === 'curated-mcp-faq-fefd5ea7eeb7') {
    const pin = '27ab2419191334dd288b05343e3a622ed615913d';
    const github = 'https://github.com/modelcontextprotocol/modelcontextprotocol/blob/' + pin + '/';
    const original = 'https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/' + pin + '/';
    const asset = '00 Учебник/Assets/Figures/curated/tool-use-2026-11-09/mcp-simple-diagram.png';
    const license = 'site/public/licenses/mcp-27ab2419-MIT.txt';
    const digest = 'fefd5ea7eeb7289f1d9f00552ce798f2298b1f588a8733a79b9ee4c10512e7c2';
    expect(entry).toMatchObject({
      asset, source_asset: asset, commit: pin,
      source_url: github + 'docs/faqs.mdx',
      source_image_url: original + 'docs/images/mcp-simple-diagram.png',
      sha256: digest, source_asset_sha256: digest, dimensions_px: [3840, 1500],
      license: 'MIT', license_url: github + 'LICENSE', license_evidence: license,
      license_evidence_sha256: '346e57dc40801e3d58165e97f64d45b42c4571a68a76ecb10c3c12858acecb3f',
      public_license_copy: license, derivation: 'source-raster-copy',
      provenance_confirmation: 'pinned-upstream-repository',
      rights_status: 'source-license-verified', metadata_status: 'verified',
    });
    expect(sha256(resolve(repositoryRoot, asset))).toBe(digest);
    expect(sha256(resolve(repositoryRoot, license))).toBe(entry.license_evidence_sha256);
    expect(pngDimensions(resolve(repositoryRoot, asset))).toEqual([3840, 1500]);
    expect(entry.crop_box_px).toBeUndefined();
    expectDecodableImage(resolve(repositoryRoot, asset));
    return;
  }
  if (entry.id === 'curated-vlm2026-cs231n-winoground-d72312341628') {
    expect(entry).toMatchObject({
      asset: '00 Учебник/Assets/Figures/curated/vlm-2026/cs231n-lecture16-winoground.png',
      publication_status: 'retired', used_in: [],
      license: 'Unverified for redistribution', license_url: null,
      provenance_confirmation: 'source-linked-rights-unverified',
      rights_status: 'course-slide-republication-unverified',
      sha256: '421bc90192b099809f93bbc97de2bbaf5948b747d1ecba83cfccd66e233e7e93',
    });
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  const metricPrefix = '00 Учебник/Assets/Figures/curated/selection-metric-2026-11-04/';
  if (entry.asset.startsWith(metricPrefix)) {
    const cases = [
      ['mooc-kmeans-not-scaled.svg', 'clustering_quiz_kmeans_not_scaled.svg',
        '04c1dd673b6b7690a0bc9e7c06358a43ec7b38e64c7cac401180bb9cf37ec2c8'],
      ['mooc-kmeans-scaled.svg', 'clustering_quiz_kmeans_scaled.svg',
        'bd2aae6c6eca8e5951324b268c312dc52263a697e7de1d7b3040bc3df2310694'],
    ];
    const selected = cases.find(([file]) => entry.asset === metricPrefix + file);
    expect(selected).toBeDefined();
    const [, original, digest] = selected!;
    const pin = '0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6';
    const upstream = 'https://github.com/INRIA/scikit-learn-mooc/blob/' + pin + '/';
    const page = 'jupyter-book/clustering/clustering_quiz_m4_01.md';
    const license = 'site/public/licenses/scikit-learn-mooc-0cc70c41-CC-BY-4.0.txt';
    const licenseHash = 'f5b745ef98087f531e719ee8ca6a96809444573ecc7173c6fa68eaad39b3cc3f';
    expect(entry.id).toBe('curated-mooc-scaling-' + digest.slice(0, 12));
    expect(entry.author).toBe('INRIA, scikit-learn MOOC contributors');
    expect(entry.commit).toBe(pin);
    expect(entry.source_url).toBe(upstream + page);
    expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/' + pin + '/figures/' + original);
    expect(entry.source_asset).toBe('Assets/Sources/Scikit-learn MOOC/figures/' + original);
    expect(entry.source_page_snapshot).toBe('05 Источники/Courses/Scikit-learn MOOC/' + page + '.md');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe(upstream + 'LICENSE');
    expect(entry.license_evidence).toBe(license);
    expect(entry.public_license_copy).toBe(license);
    expect(entry.license_evidence_sha256).toBe(licenseHash);
    expect(sha256(resolve(repositoryRoot, license))).toBe(licenseHash);
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.source_asset_sha256).toBe(digest);
    expect(entry.sha256).toBe(digest);
    expect(entry.source_viewbox).toEqual([0, 0, 375.6672, 325.18365]);
    expect(entry.crop_box_px).toBeUndefined();
    requireNonEmptyString(entry.modifications, 'modifications');
    for (const file of [entry.asset, entry.source_asset!])
      expect(sha256(resolve(repositoryRoot, file))).toBe(digest);
    const svg = readFileSync(resolve(repositoryRoot, entry.asset), 'utf8');
    expect(svg).toContain('viewBox="0 0 375.6672 325.18365"');
    expect(readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8')).toContain(original);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/sequence-state-2026-11-03/')) {
    const figures=JSON.parse(readFileSync(resolve(repositoryRoot,
      'publishing/tools/sequence-state-figures.json'),'utf8'));
    const spec=figures.find((x:{asset:string})=>x.asset===entry.asset);
    expect(spec).toBeDefined();
    const expected={
      author:'Dustin Wang et al.', source_url:spec.source_url,
      source_asset:spec.source,source_sha256:spec.source_sha256,
      source_page:spec.page,source_object:spec.object,license:spec.license,
      license_url:spec.license_url,license_evidence:spec.license_evidence,
      license_evidence_sha256:spec.license_evidence_sha256,sha256:spec.sha256,
      dimensions_px:spec.dimensions,dpi:spec.dpi,crop_box_px:spec.crop_pixels,
      derivation:'pdf-page-render-crop',metadata_status:'verified',
      rights_status:'verified-open-license',
      provenance_confirmation:'pinned-versioned-arxiv-pdf-and-license-page',
    };
    for(const [key,value] of Object.entries(expected))
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    for(const [path,hash] of [[spec.source,spec.source_sha256],
      [spec.license_evidence,spec.license_evidence_sha256],[spec.asset,spec.sha256]])
      expect(sha256(resolve(repositoryRoot,path))).toBe(hash);
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(spec.dimensions);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/gpu-copy-2026-11-04/')) {
    const manifest=JSON.parse(readFileSync(resolve(repositoryRoot,
      'publishing/tools/gpu-copy-source-figures.json'),'utf8'));
    expect(manifest.pin).toBe('5672cd98a7671eda4d9122b10f8eec580f74d03e');
    const file=entry.asset.split('/').at(-1)!;
    const spec=manifest.files.find((x:{name:string;dimensions:number[]|null})=>x.name===file&&x.dimensions);
    expect(spec).toBeDefined();
    expect(entry.id).toBe('curated-gpu-copy-'+spec.sha256.slice(0,12));
    expect(entry.sha256).toBe(spec.sha256);
    expect(entry.source_asset_sha256).toBe(spec.sha256);
    expect(entry.source_asset).toBe(entry.asset);
    expect(entry.author).toBe('Vincent Moens; PyTorch contributors');
    expect(entry.source_url).toBe('https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html');
    expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/pytorch/tutorials/'+manifest.pin+'/_static/img/pinmem/'+file);
    expect(entry.commit).toBe(manifest.pin);
    expect(entry.license).toBe('BSD-3-Clause');
    expect(entry.license_url).toBe('https://github.com/pytorch/tutorials/blob/'+manifest.pin+'/LICENSE');
    expect(entry.license_evidence).toBe('site/public/licenses/pytorch-tutorials-5672cd98-BSD-3-Clause.txt');
    expect(entry.license_evidence_sha256).toBe('5703fcbe2d5dfcc48d486d26e4b9d2e48255c24602e1bd0270b60b8978116f86');
    expect(sha256(resolve(repositoryRoot,entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.dimensions_px).toEqual(spec.dimensions);
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(spec.sha256);
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(spec.dimensions);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (/^curated-cs231n-(mlp|depth|regions)-00000000008[abc]$/.test(entry.id ?? '')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/mlp-2026-11-03/assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.id===entry.id));
    const specs = {
      'cs231n-mlp-layered.jpeg':['649d51c291077a114e32cdbd282b49841124383ff63a91d612a4d0d2310641c8','neural_net2.jpeg',791,388],
      'cs231n-depth-width.jpeg':['ce7b76035580f3b6d87aac3089ba87967f29423d1e762b04a1cbd463d5d6b6e5','layer_sizes.jpeg',1031,366],
      'cs231n-decision-regions.jpeg':['94aadb3b92581a7155ee44b0e17552d0c4c40ec671e1b92b6884f05b1e5ba501','reg_strengths.jpeg',1077,394],
    } as Record<string,[string,string,number,number]>;
    const spec=specs[entry.asset.split('/').at(-1)!];
    expect(spec).toBeDefined();
    expect(entry.sha256).toBe(spec[0]);
    expect(entry.source_asset).toBe(entry.asset);
    expect(entry.source_asset_sha256).toBe(spec[0]);
    expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/cs231n/cs231n.github.io/8ec57ccfc32379ae4b37c45de949af359673f92f/assets/nn1/'+spec[1]);
    expect(entry.commit).toBe('8ec57ccfc32379ae4b37c45de949af359673f92f');
    expect(entry.license).toBe('MIT');
    expect(entry.license_evidence_sha256).toBe('85b365f07b4cfc5d678c548bc41965bc3bc782106b8903bd3b876d9d49a4270d');
    expect(entry.source_page_sha256).toBe('28196e335053230ac9f1fff728c8baa089e05814982d950ffc6339462828f0f3');
    expect(entry.dimensions_px).toEqual([spec[2],spec[3]]);
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.derivation).toBe('source-image-copy');
    for(const [path,hash] of [[entry.asset,spec[0]],
      [entry.license_evidence!,entry.license_evidence_sha256!],
      [entry.source_page_snapshot!,entry.source_page_sha256!]])
      expect(sha256(resolve(repositoryRoot,path))).toBe(hash);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/probability-native-2026-10-30/')) {
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/probability-native-2026-10-30/assets.json'), 'utf8'));
    expect(rows).toHaveLength(2);
    expect(entry).toEqual(rows.find((row: CuratedAsset) => row.asset === entry.asset));
    const specs: Record<string, string> = {
      'd2l-density-interval.svg': '35d85344b8bd9faf7cda96b89bfd235754990ec381f3fdb0ca9d30a812bac27f',
      'd2l-marginal.svg': 'cda896c59fd8dfb81f4d5d03fd392142da25b331378e1da668a0088cbd44dd51'
    };
    const file = entry.asset.split('/').at(-1)!;
    expect(specs[file]).toBeDefined();
    expect(entry.sha256).toBe(specs[file]);
    expect(entry.source_asset).toBe(entry.asset);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.source_code_commit).toBe('23d7a5aecceee57d1292c56e90cce307f183bb0a');
    expect(entry.source_code_sha256).toBe('506e48d177d6753b65a671702f4e85422d0501496a085728960799ec4bd1e43f');
    expect(entry.license).toBe('CC BY-SA 4.0');
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.crop_box_px).toBeUndefined();
    for (const [path, digest] of [[entry.asset, specs[file]],
      [entry.source_code, entry.source_code_sha256],
      [entry.license_evidence, 'a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c']])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(digest);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/derivative-native-2026-10-30/')) {
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/derivative-native-2026-10-30/assets.json'), 'utf8'));
    expect(rows).toHaveLength(2);
    expect(entry).toEqual(rows.find((row: CuratedAsset) => row.asset === entry.asset));
    const specs: Record<string, [string, string]> = {
      'd2l-function-tangent.svg': [
        'fcb32c3fa05846549df55720dc958478f9260507c4d1377a4b31d3940cf9bce2',
        'a325c626fd77f5be37d32579cd0a5cf4a75c9179c995bce08baa487e05026419'],
      'd2l-chain-paths.svg': [
        '897746e112a76eb09af174d5f79fbb49a936142567029d92d29a84d602e5cda6',
        '2064011aea85df2626ac6fba36db283b79c5d60ebdab1090cf62a76ecee8014d']
    };
    const file = entry.asset.split('/').at(-1)!;
    expect(specs[file]).toBeDefined();
    expect(entry.sha256).toBe(specs[file][0]);
    expect(entry.source_code_sha256).toBe(specs[file][1]);
    expect(entry.source_asset).toBe(entry.asset);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.license).toBe('CC BY-SA 4.0');
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.crop_box_px).toBeUndefined();
    for (const [path, digest] of [[entry.asset, specs[file][0]],
      [entry.source_code, specs[file][1]],
      [entry.license_evidence, 'a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c']])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(digest);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/tensor-native-geometry-2026-10-30/')) {
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/tensor-native-geometry-2026-10-30/assets.json'), 'utf8'));
    expect(rows).toHaveLength(2);
    expect(entry).toEqual(rows.find((row: CuratedAsset) => row.asset === entry.asset));
    const specs: Record<string, string> = {
      'vec-add.svg': 'c4cafd62fbe2eaff376fb6142f4bbd78d2cb5cfc90448042db61a91a1267f384',
      'vec-angle.svg': '46cda699171505338dbd063fd66b0cbacecd95eb8db16b5bc89a7456380e9c3f'
    };
    const file = entry.asset.split('/').at(-1)!;
    expect(specs[file]).toBeDefined();
    expect(entry.sha256).toBe(specs[file]);
    expect(entry.source_asset).toBe(entry.asset);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.commit).toBe('23d7a5aecceee57d1292c56e90cce307f183bb0a');
    expect(entry.source_image_url).toBe(`https://raw.githubusercontent.com/d2l-ai/d2l-en/${entry.commit}/img/${file}`);
    expect(entry.license).toBe('CC BY-SA 4.0');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.crop_box_px).toBeUndefined();
    for (const [path, digest] of [[entry.asset, entry.sha256],
      [entry.source_code, '066c856e5860ea00a7cd49542250e4c97dc810c0a9bfbe3c1e4bf2e66d7c356e'],
      [entry.license_evidence, 'a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c']])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(digest);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/bleu-reconciliation-2026-10-30/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/bleu-reconciliation-2026-10-30/assets.json'),'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(1);
    expect(entry).toEqual(rows[0]);
    expect(entry.sha256).toBe('d917184b34ffda8ff892710a84a611d949bc23b39b383956d4053ceb7cab66ce');
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/Tiiiger/bert_score/19e7f551fe4fa43fdd07b8129ae947015b902b2d/bert_score.png');
    expect(entry.derivation).toBe('source-image-copy');
    expect(entry.license).toBe('MIT');
    for(const [path,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256],
      [entry.public_license_copy,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset)))).toEqual([2533,728]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/tot-reconciliation-2026-10-30/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/tot-reconciliation-2026-10-30/assets.json'),'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(3);
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    const specs:Record<string,[string,number,number[],number[]]>={
      'tot-creative-search.png':['61bea3f4f86b0728499a82838196f3bd8aca11d6045e180e721522b81d6cac71',7,[440,305,2130,1055],[1690,750]],
      'tot-creative-results.png':['da6f37a5e39f90a3336f40197b6f47fd80c9053412381ea47112730cdda41231',7,[440,1075,1490,1595],[1050,520]],
      'tot-crossword-dfs.png':['2c692a2260acee60bcf348b50c14445dcbaaf9da027c8e13901d19b34adeb37d',8,[440,325,2130,1080],[1690,755]],
    };
    const spec=specs[entry.asset.split('/').at(-1)!];
    expect(spec).toBeDefined();
    expect(entry.sha256).toBe(spec[0]);
    expect(entry.source_page).toBe(spec[1]);
    expect(entry.crop_box_px).toEqual(spec[2]);
    expect(entry.render_page_size_px).toEqual([2550,3300]);
    expect(entry.source_url).toBe('https://arxiv.org/abs/2305.10601v2');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.source_asset_sha256).toBe('79c5237e3f63953a73f2b0d6894327702ee1f7e981450c251bb1b5cb4f8d7b8f');
    expect(entry.license_evidence_sha256).toBe('9da03295d951ff9e2df517c41c438b16711c86aab22f0f137c8a9e8704d6995b');
    for(const [path,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset)))).toEqual(spec[3]);
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8'))
      .toContain('creativecommons.org/licenses/by/4.0/');
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/agent-reconciliation-2026-10-30/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/agent-reconciliation-2026-10-30/assets.json'),'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(2);
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    const sampling=entry.asset.endsWith('/react-cot-sampling.png');
    expect(entry.sha256).toBe(sampling
      ? '23c9dbf17b79eca8bd8f3cfcb34d623bd70ca02fb025c7ab3135df6bae246989'
      : '4be61fc97b54ecf9a2bf56ef80fa069bddbe3071c82cce0fa300c642e6791706');
    expect(entry.source_url).toBe('https://arxiv.org/abs/2210.03629v3');
    expect(entry.source_page).toBe(sampling ? 5 : 7);
    expect(entry.crop_box_px).toEqual(sampling ? [1160,420,2120,1018] : [440,340,2120,940]);
    expect(entry.render_page_size_px).toEqual([2550,3300]);
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.source_asset_sha256).toBe('f285b0971ae4a790e402fb93966bed3adde2cf0a04977d08b2b40d6ab0cace69');
    expect(entry.license_evidence_sha256).toBe('dc39550159e139ddd3e88224a75bd153ba71d55684dddb4bcf2d74262f93a0a5');
    for(const [path,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual(sampling ? [960,598] : [1680,600]);
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8'))
      .toContain('creativecommons.org/licenses/by/4.0/');
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/nanogpt-2026-10-30/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/nanogpt-2026-10-30/assets.json'),'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(1);
    expect(entry).toEqual(rows[0]);
    expect(entry.sha256).toBe('50bb25d32eea8873f6fac6955366a2be607af6ac051972be2c84646a7eff83ae');
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/karpathy/nanoGPT/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/assets/gpt2_124M_loss.png');
    expect(entry.license).toBe('MIT');
    expect(entry.derivation).toBe('source-image-copy');
    for(const [path,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256],
      [entry.public_license_copy,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual([1206,614]);
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8'))
      .toContain('Copyright (c) 2022 Andrej Karpathy');
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/synthetic-recipes-2026-10-29/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/synthetic-recipes-2026-10-29/assets.json'),'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(3);
    const row=rows.find(candidate=>candidate.asset===entry.asset);
    expect(row).toBeDefined();
    expect(entry).toEqual(row);
    expect(entry.source_asset).toMatch(/^05 Источники\/Originals\/synthetic-recipes-2026-10-29\/(?:phi1-data-ablation|phi4-repetitions|nemotron4-prompts)\.png$/);
    expect(entry.source_image_url).toMatch(/^https:\/\/arxiv.org\/html\/(?:2306\.11644v2\/figures\/new_data_results|2412\.08905v1\/figures\/mmlu_synth_epochs|2406\.11704v1\/image\/synthetic-prompts)\.png$/);
    expect(entry.derivation).toBe('source-image-copy');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.rights_status).toBe('source-license-verified');
    for(const [path,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot,entry.asset)))
      .toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset)));
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual(entry.dimensions_px);
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8'))
      .toContain('License: CC BY 4.0');
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/corpus-releases-2026-10-29/')) {
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/corpus-releases-2026-10-29/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(1);
    expect(entry).toEqual(rows[0]);
    expect(entry.source_url).toBe('https://arxiv.org/abs/2303.03915v1');
    expect(entry.source_image_url).toBe('https://arxiv.org/html/2303.03915v1/figures/oscar_filtered.png');
    expect(entry.sha256).toBe('eb44fc2443ccac7e3b57a138ba64f626b82d0330261c4e22da64e99042a2309e');
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.derivation).toBe('source-raster-copy');
    for (const [file,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,file!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot,entry.asset)))
      .toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual([1552,279]);
    const page=readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8');
    expect(page).toContain('License: CC BY 4.0');
    expect(page).toContain('id="S3.F3"');
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/bert-objectives-2026-10-29/')
    || entry.id?.startsWith('curated-bert-objectives-')) {
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/bert-objectives-2026-10-29/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(2);
    expect(entry).toEqual(rows.find(row => row.asset === entry.asset));
    const first = entry.asset.endsWith('/spanbert-boundary-objective.png');
    expect(entry.asset).toBe('00 Учебник/Assets/Figures/curated/bert-objectives-2026-10-29/'
      + (first ? 'spanbert-boundary-objective.png' : 'spanbert-span-lengths.png'));
    expect(entry.sha256).toBe(first
      ? '1add71c0a3c2291faae8e87466905d82e129c89a3fe874b9a22b7536bb2b95e9'
      : 'be2494e1a024c6e7de135f0de896e1a2f2fc3108776a3fa51292e20f2410c7ac');
    expect(entry.source_asset).toBe('05 Источники/Originals/bert-objectives-2026-10-29/spanbert-author.pdf');
    expect(entry.source_asset_sha256).toBe('86bad382bd3ba3bb1cae16636da6867a5846e10f7d881516af0e1ccd67a5925b');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.source_page).toBe(first ? 2 : 3);
    expect(entry.render_dpi).toBe(288);
    expect(entry.render_page_size_px).toEqual([2380, 3368]);
    expect(entry.crop_box_px).toEqual(first ? [280,180,2120,1150] : [1210,205,2120,1090]);
    expect(entry.dimensions_px).toEqual(first ? [1840,970] : [910,885]);
    for (const [file,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,file!))).toBe(digest);
    const license = execFileSync('pdftotext', ['-f','1','-l','1',
      resolve(repositoryRoot,entry.license_evidence!),'-'], {encoding:'utf8'});
    expect(license).toContain('Distributed under a CC-BY 4.0 license.');
    const box = entry.crop_box_px as number[];
    const originalCrop = execFileSync('pdftoppm', [
      '-f',String(entry.source_page),'-l',String(entry.source_page),'-r','288',
      '-x',String(box[0]),'-y',String(box[1]),'-W',String(box[2]-box[0]),
      '-H',String(box[3]-box[1]),'-singlefile','-png',
      resolve(repositoryRoot,entry.source_asset!),
    ], {env:{...process.env,LC_ALL:'C',LANG:'C'},maxBuffer:4*1024*1024});
    expect(originalCrop).toEqual(readFileSync(resolve(repositoryRoot,entry.asset)));
    expect(validateRenderedPngBytes(originalCrop)).toEqual(entry.dimensions_px);
    return;
  }
  const publicationProvenance = [
    {
      "asset": "00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_self-attention_visualization.png",
      "hash": "5f85f00a27e48125cd9445f562943b2ca8e7a73d61e9de381fc106a21463e5c4",
      "size": [
        437,
        413
      ],
      "source": "05 Источники/Originals/publication-asset-provenance-2026-10-29/alammar-self-attention.png",
      "sourceUrl": "https://jalammar.github.io/illustrated-transformer/",
      "imageUrl": "https://jalammar.github.io/images/t/transformer_self-attention_visualization.png",
      "license": "CC BY-NC-SA 4.0",
      "evidence": "05 Источники/Originals/self-attention-2026-10-21/alammar-illustrated-transformer.html",
      "evidenceHash": "375d8acf1096832ddfff288d93ff0ace405b6a4897bb96e280e6d71f3d353556"
    },
    {
      "asset": "00 Учебник/Assets/Figures/curated/mixtral-of-experts/smoe-layer.png",
      "hash": "1347eb75040f1144f7ea5820cae7b4bd60c49b6bf351fb1c9e77b4c20e642c46",
      "size": [
        2116,
        968
      ],
      "source": "05 Источники/Originals/publication-asset-provenance-2026-10-29/mixtral-smoe.png",
      "sourceUrl": "https://arxiv.org/abs/2401.04088v1",
      "imageUrl": "https://arxiv.org/html/2401.04088v1/images/smoe.png",
      "license": "CC BY 4.0",
      "evidence": "05 Источники/Originals/publication-asset-provenance-2026-10-29/mixtral-arxiv-v1.html",
      "evidenceHash": "f64f17740fbeca1bad2406c8b2402c24c019011deb0c1c451eef2beecad6a226"
    },
    {
      "asset": "00 Учебник/Assets/Figures/curated/dpo/fig1.png",
      "hash": "af51c080272561bcb736aef8b56c1ef220465f89b8ad3bf045014922941e89c4",
      "size": [
        2380,
        478
      ],
      "source": "05 Источники/Originals/publication-asset-provenance-2026-10-29/dpo-teaser.png",
      "sourceUrl": "https://arxiv.org/abs/2305.18290v3",
      "imageUrl": "https://arxiv.org/html/2305.18290v3/figures/diagrams/teaser.png",
      "license": "CC BY 4.0",
      "evidence": "05 Источники/Originals/publication-asset-provenance-2026-10-29/dpo-arxiv-v3.html",
      "evidenceHash": "44db9b2b1c0c8bb664e2c8482b57d7eedfe7afc9321eb644fb6309f8906fe795"
    },
    {
      "asset": "00 Учебник/Assets/Figures/curated/instructgpt/reward-model-hf.png",
      "hash": "30107284e8d7b9b1bf859363656dd01439024457d063ae22f8bc3cf90393cec9",
      "size": [
        2080,
        1571
      ],
      "source": "05 Источники/Originals/publication-asset-provenance-2026-10-29/hf-reward-model.png",
      "sourceUrl": "https://huggingface.co/blog/rlhf",
      "imageUrl": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/8d04c5b37984c76e594293aab2d610490dae8ec4/blog/rlhf/reward-model.png",
      "license": "CC BY-NC-SA 4.0",
      "evidence": "05 Источники/Originals/publication-asset-provenance-2026-10-29/hf-documentation-README.md",
      "evidenceHash": "5e6a73b7ff47813e90e2ed21d14807256b2e89fa55e5075ec69fdf01cde801ce"
    },
    {
      "asset": "00 Учебник/Assets/Figures/curated/source-first-7-14/cs231n-local-connectivity.jpeg",
      "hash": "f41da3fd33d2ed75aaaae4530359bb172ff338ea3ac8bb9f1d6499676db3fba7",
      "size": [
        569,
        202
      ],
      "source": "05 Источники/Originals/publication-asset-provenance-2026-10-29/cs231n-cnn.jpeg",
      "sourceUrl": "https://github.com/cs231n/cs231n.github.io/blob/8ec57ccfc32379ae4b37c45de949af359673f92f/assets/cnn/cnn.jpeg",
      "imageUrl": "https://raw.githubusercontent.com/cs231n/cs231n.github.io/8ec57ccfc32379ae4b37c45de949af359673f92f/assets/cnn/cnn.jpeg",
      "license": "MIT",
      "evidence": "05 Источники/Originals/publication-asset-provenance-2026-10-29/cs231n-LICENSE.txt",
      "evidenceHash": "85b365f07b4cfc5d678c548bc41965bc3bc782106b8903bd3b876d9d49a4270d"
    },
    {
      "asset": "00 Учебник/Assets/Figures/curated/source-first-7-14/cs231n-stride-padding.jpeg",
      "hash": "5b5f71b55bac0ccdad72189c12aee5c5a64f75ef808dc1338230740d629aa1ab",
      "size": [
        861,
        172
      ],
      "source": "05 Источники/Originals/publication-asset-provenance-2026-10-29/cs231n-stride.jpeg",
      "sourceUrl": "https://github.com/cs231n/cs231n.github.io/blob/8ec57ccfc32379ae4b37c45de949af359673f92f/assets/cnn/stride.jpeg",
      "imageUrl": "https://raw.githubusercontent.com/cs231n/cs231n.github.io/8ec57ccfc32379ae4b37c45de949af359673f92f/assets/cnn/stride.jpeg",
      "license": "MIT",
      "evidence": "05 Источники/Originals/publication-asset-provenance-2026-10-29/cs231n-LICENSE.txt",
      "evidenceHash": "85b365f07b4cfc5d678c548bc41965bc3bc782106b8903bd3b876d9d49a4270d"
    }
  ];
  const confirmed = publicationProvenance.find(spec => spec.asset === entry.asset);
  if (confirmed) {
    const packet = '05 Источники/Originals/publication-asset-provenance-2026-10-29/';
    const manifest = JSON.parse(readFileSync(resolve(repositoryRoot, packet + 'verified-assets.json'), 'utf8'));
    expect(manifest.assets).toHaveLength(6);
    expect(entry).toEqual(manifest.assets.find((row: CuratedAsset) => row.asset === entry.asset));
    expect(entry.sha256).toBe(confirmed.hash);
    expect(entry.source_asset_sha256).toBe(confirmed.hash);
    expect(entry.source_asset).toBe(confirmed.source);
    expect(entry.source_url).toBe(confirmed.sourceUrl);
    expect(entry.source_image_url).toBe(confirmed.imageUrl);
    expect(entry.dimensions_px).toEqual(confirmed.size);
    expect(entry.license).toBe(confirmed.license);
    expect(entry.license_evidence).toBe(confirmed.evidence);
    expect(entry.license_evidence_sha256).toBe(confirmed.evidenceHash);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.crop_box_px).toBeUndefined();
    requireNonEmptyString(entry.author, 'identified primary author');
    requireNonEmptyString(entry.modifications, 'original byte-copy description');
    for (const [file, digest] of [[entry.asset, entry.sha256],
      [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(digest);
    const published = readFileSync(resolve(repositoryRoot, entry.asset));
    expect(published).toEqual(readFileSync(resolve(repositoryRoot, entry.source_asset!)));
    expect(published.length).toBe(entry.bytes);
    const evidence = readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8');
    if (entry.license === 'MIT') {
      expect(entry.commit).toBe('8ec57ccfc32379ae4b37c45de949af359673f92f');
      expect(entry.license_url).toBe('https://github.com/cs231n/cs231n.github.io/blob/' + entry.commit + '/LICENSE');
      expect(evidence).toContain('Copyright (c) 2015 Andrej Karpathy');
      expect(evidence).toContain('Permission is hereby granted, free of charge');
      expect(evidence).toContain('The above copyright notice and this permission notice shall be included');
    } else {
      const cc = entry.license === 'CC BY 4.0' ? 'by/4.0/' : 'by-nc-sa/4.0/';
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/' + cc);
      if (entry.asset.endsWith('/reward-model-hf.png')) {
        expect(entry.commit).toBe('8d04c5b37984c76e594293aab2d610490dae8ec4');
        expect(evidence).toContain('license: cc-by-nc-sa-4.0');
      } else expect(evidence).toContain('creativecommons.org/licenses/' + cc);
      expect(validateRenderedPngBytes(published)).toEqual(confirmed.size);
    }
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if ([
    '00 Учебник/Assets/Figures/curated/reference-source-first/helm-taxonomy-figure2.png',
    '00 Учебник/Assets/Figures/curated/consolidation-2026-10-29/constitutional-ai-overview.png'
  ].includes(entry.asset)) {
    const packet = '05 Источники/Originals/training-consolidation-2026-10-29/';
    const manifest = JSON.parse(readFileSync(resolve(repositoryRoot, packet + 'assets.json'), 'utf8'));
    expect(manifest.assets).toHaveLength(2);
    expect(entry).toEqual(manifest.assets.find((row: CuratedAsset) => row.asset === entry.asset));
    const helm = entry.asset.endsWith('helm-taxonomy-figure2.png');
    const paper = helm ? '2211.09110v2' : '2212.08073v1';
    const outputHash = helm
      ? '9288b3710949b979f737b88274135b28836831df44068ac4ddccac44ab229ee8'
      : '43db0cfc079e1864f8aa5358962a290090c6980b77701665bebd91f1e4d54ea4';
    const sourceHash = helm
      ? '30970bda8958a1a8adf6a0a82feef9728331840288b7d2b49368bac87fd25f18'
      : outputHash;
    const snapshotHash = helm
      ? 'acfa5fdf9d40eeda2e0a8db8e6d0886631e53157794f11faa6efef9a3f03fc85'
      : '2f175dac8902db65a5c0a7cd280109fa91e0d3c1d129e49edb9ab439edc3c97d';
    const snapshot = packet + (helm ? 'helm-arxiv-v2.html' : 'cai-arxiv-v1.html');
    expect(entry.id).toBe(helm ? 'curated-helm-taxonomy-0000000000c6' : 'curated-cai-overview-43db0cfc079e');
    expect(entry.author).toBe(helm ? 'Percy Liang et al.' : 'Yuntao Bai et al.');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.source_url).toBe('https://arxiv.org/abs/' + paper);
    expect(entry.source_image_url).toBe(helm
      ? 'https://arxiv.org/pdf/2211.09110v2#page=3'
      : 'https://arxiv.org/html/2212.08073v1/CAI-Overview.png');
    expect(entry.source_asset).toBe(packet + (helm ? 'helm-v2.pdf' : 'CAI-Overview.png'));
    expect(entry.source_asset_sha256).toBe(sourceHash);
    expect(entry.sha256).toBe(outputHash);
    expect(entry.source_page_snapshot).toBe(snapshot);
    expect(entry.source_page_sha256).toBe(snapshotHash);
    expect(entry.license_evidence).toBe(snapshot);
    expect(entry.license_evidence_sha256).toBe(snapshotHash);
    expect(entry.dimensions_px).toEqual(helm ? [2010, 1005] : [2025, 844]);
    expect(entry.bytes).toBe(helm ? 228163 : 140646);
    expect(entry.used_in).toEqual([helm
      ? '00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md'
      : '00 Учебник/12 Post-training и Alignment/00a Цели alignment — HHH, Constitutional AI и RLAIF.md']);
    requireNonEmptyString(entry.modifications, 'complete original-figure derivation');
    for (const [file, digest] of [[entry.asset, entry.sha256],
      [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(digest);
    const snapshotText = readFileSync(resolve(repositoryRoot, snapshot), 'utf8');
    expect(snapshotText).toContain('/pdf/' + paper);
    expect(snapshotText).toMatch(/<div class="abs-license"><a href="https?:\/\/creativecommons\.org\/licenses\/by\/4\.0\/"[^>]*title="Rights to this article"/);
    const published = readFileSync(resolve(repositoryRoot, entry.asset));
    expect(published.length).toBe(entry.bytes);
    expect(validateRenderedPngBytes(published)).toEqual(entry.dimensions_px);
    for (const chapter of entry.used_in!)
      expect(readFileSync(resolve(repositoryRoot, chapter), 'utf8')).toContain(entry.asset);
    if (helm) {
      expect(entry.derivation).toBe('pdf-page-render-crop');
      expect(entry.source_page).toBe(3);
      expect(entry.render_dpi).toBe(300);
      expect(entry.render_page_size_px).toEqual([2550, 3300]);
      expect(entry.crop_box_px).toEqual([270, 300, 2280, 1305]);
      const rendered = execFileSync('pdftoppm', ['-f', '3', '-l', '3', '-r', '300',
        '-x', '270', '-y', '300', '-W', '2010', '-H', '1005', '-singlefile', '-png',
        resolve(repositoryRoot, entry.source_asset!)],
        { env: { ...process.env, LC_ALL: 'C', LANG: 'C' }, maxBuffer: 4 * 1024 * 1024 });
      expect(rendered).toEqual(published);
    } else {
      expect(entry.derivation).toBe('source-raster-copy');
      expect(entry.source_page).toBeUndefined();
      expect(entry.render_dpi).toBeUndefined();
      expect(entry.render_page_size_px).toBeUndefined();
      expect(entry.crop_box_px).toBeUndefined();
      expect(readFileSync(resolve(repositoryRoot, entry.source_asset!))).toEqual(published);
    }
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/cs336-residuals-2026-10-29/')) {
    const manifest=JSON.parse(readFileSync(resolve(repositoryRoot,'docs/audit/2026-10-29-cs336-residual-assets.json'),'utf8'));
    const overlay=JSON.parse(readFileSync(resolve(repositoryRoot,'docs/audit/2026-10-29-cs336-asset-evidence.json'),'utf8')).assets;
    const spec=manifest.assets.find((row:CuratedAsset)=>row.asset===entry.asset);
    expect(spec).toBeDefined();
    expect(entry).toEqual({...spec,...(overlay[entry.id!]??{})});
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(entry.sha256);
    if(entry.source_asset) {
      const source=readFileSync(resolve(repositoryRoot,entry.source_asset));
      if(entry.source_asset_sha256)expect(sha256(resolve(repositoryRoot,entry.source_asset))).toBe(entry.source_asset_sha256);
      if(entry.derivation==='source-raster-copy')expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(source);
    }
    if(entry.asset.endsWith('.png'))expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset)))).toEqual(entry.dimensions_px);
    else expectDecodableImage(resolve(repositoryRoot,entry.asset));
    if(entry.license_evidence)expect(sha256(resolve(repositoryRoot,entry.license_evidence))).toBe(entry.license_evidence_sha256);
    if(entry.source_page_snapshot)expect(sha256(resolve(repositoryRoot,entry.source_page_snapshot))).toBe(entry.source_page_sha256);
    expect(entry.metadata_status).toBe('verified');
    if(entry.id?.includes('air-taxonomy'))expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8')).toContain('License: CC BY 4.0');
    if(entry.id?.includes('-v4-'))expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8')).toContain('MIT');
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/serving-decisions-2026-10-29/')) {
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,'00 Учебник/Assets/Figures/curated/serving-decisions-2026-10-29/assets.json'),'utf8'));
    expect(entry).toEqual(specs.find((row:CuratedAsset)=>row.asset===entry.asset));
    for(const [path,digest]of[[entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256],[entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset)))).toEqual([5190,2077]);
    expect(entry.derivation).toBe('source-image-copy');
    expect(entry.license).toBe('CC BY-NC-SA 4.0');
    return;
  }
  if (entry.id === 'curated-qwen-next-block-original-07f983ce9648') {
    expect(entry.asset).toBe('00 Учебник/Assets/Figures/curated/stanford-cs336-2026/architectures/qwen-next-block-original.jpg');
    expect(entry.provenance_confirmation).toBe('cs336-qwen-source-snapshot');
    expect(entry.derivation).toBe('pdf-embedded-jpeg-copy');
    expect(entry.metadata_status).toBe('needs-confirmation');
    expect(entry.rights_status).toBe('third-party-figure-license-unverified');
    expect(entry.source_page).toBe(10);
    expect(entry.source_pdf_object).toBe(69);
    expect(entry.source_asset).toBe('05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/lecture_04.pdf');
    expect(entry.source_asset_sha256).toBe('922da01cee70ed9dc9eef13d8116f1d6f45d663c12e632b3e80954c783bd3ce5');
    expect(entry.sha256).toBe('07f983ce96489f9c8fe6d3ea9fd9101bd02421cc65a4fb6ee48197949ea138ff');
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.render_dpi).toBeUndefined();
    return;
  }
  if (entry.id?.startsWith('curated-gpt1-source-panel-')) {
    const specs = {
      'gpt1-figure1-block.png': { page: 4, crop: [550, 340, 980, 1140], sha: 'a6d995e45713f76cec87c12e82733766ad33526a4268a59fa91aeddeb43eb1ce' },
      'gpt1-figure2-transfer.png': { page: 7, crop: [535, 2210, 1510, 3055], sha: '7101483f51465dc8503b01094e5aa4f5d522033f00bdb4d5ece784cb03eb6876' },
      'gpt1-figure2-zero-shot.png': { page: 7, crop: [1525, 2210, 2485, 3055], sha: '8e17d3e06bcd21fd7d3b4340d5a072f456f1c0f670ef86bce10febf59623ee26' },
    };
    const name = entry.asset.split('/').at(-1)! as keyof typeof specs;
    const spec = specs[name];
    expect(spec).toBeTruthy();
    expect(entry.asset).toBe('00 Учебник/Assets/Figures/curated/source-audit-25-31/' + name);
    expect(entry.provenance_confirmation).toBe('gpt1-source-snapshot');
    expect(entry.metadata_status).toBe('needs-confirmation');
    expect(entry.rights_status).toBe('paper-figure-license-unverified');
    expect(entry.license).toBe('unverified; inherited paper figure, source verified separately');
    expect(entry.source_asset).toBe('05 Источники/Originals/full-transformer-2026-10-21/gpt1.pdf');
    expect(entry.source_asset_sha256).toBe('eb81a65e0856bd38e855e1f1de6ee12e7c3eb92ae05c8270d345a475beb631d0');
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(entry.source_page).toBe(spec.page);
    expect(entry.render_dpi).toBe(360);
    expect(entry.render_page_size_px).toEqual([3060, 3960]);
    expect(entry.crop_box_px).toEqual(spec.crop);
    expect(entry.derivation).toBe('pdf-page-render-panel');
    expect(entry.sha256).toBe(spec.sha);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(spec.sha);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([
      spec.crop[2] - spec.crop[0], spec.crop[3] - spec.crop[1],
    ]);
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/sim-to-real-2026-10-28/')) {
    const prefix='00 Учебник/Assets/Figures/curated/sim-to-real-2026-10-28/';
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,prefix+'assets.json'),'utf8'));
    expect(specs).toHaveLength(7);
    const spec=specs.find((row:CuratedAsset)=>row.asset===entry.asset);
    expect(spec).toBeTruthy();
    expect(entry).toEqual(spec);
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    if(entry.asset.endsWith('.jpg')){
      const allowed=['camera_sampling_rendering_0.jpg','camera_sampling_rendering_1.jpg',
        'material_randomizer_rendering.jpg','material_randomizer_rendering_switched.jpg',
        'semantic_segmentation_rendering_0.jpg','semantic_segmentation_rendering_1.jpg'];
      expect(allowed).toContain(entry.asset.slice(prefix.length));
      expect(entry.commit).toBe('76bd8b82ee19489843578ff045298642197bf883');
      expect(entry.license).toBe('GPL-3.0');
      expect(entry.derivation).toBe('source-jpeg-copy');
      expect(readFileSync(resolve(repositoryRoot,entry.asset)))
        .toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset!)));
      expectDecodableImage(resolve(repositoryRoot,entry.asset));
      expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8'))
        .toContain('Version 3, 29 June 2007');
    }else{
      expect(entry.asset).toBe(prefix+'dann-architecture.png');
      expect(entry.derivation).toBe('pdf-page-render-crop');
      expect(entry.license).toBe('CC BY (version unspecified in JMLR publication policy)');
      expect(entry.license_url).toBe('https://jmlr.org/author-info.html#publication');
      expect(entry.source_page).toBe(12);
      expect(entry.crop_box_px).toEqual([640,470,2840,1410]);
      expect(entry.render_dpi).toBe(400);
      const rendered=execFileSync('pdftoppm',['-f','12','-l','12','-r','400',
        '-x','640','-y','470','-W','2200','-H','940','-singlefile','-png',
        resolve(repositoryRoot,entry.source_asset!)],
        {env:{...process.env,LC_ALL:'C',LANG:'C'},maxBuffer:4*1024*1024});
      expect(rendered).toEqual(readFileSync(resolve(repositoryRoot,entry.asset)));
      expect(validateRenderedPngBytes(rendered)).toEqual([2200,940]);
    }
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/tensor-factorization-2026-10-29/')) {
    const [spec]=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/tensor-factorization-2026-10-29/assets.json'),'utf8'));
    expect(entry).toEqual(spec);
    expect(entry.source_asset).toBe('Assets/Sources/Harvard ML Systems/vol1/model_compression/46faaa58f69df0c176f0fcdc34f06a318aadafb2.svg');
    expect(entry.sha256).toBe('01e77b9b1e4e3fb5a041dbfdae5822a4af04864548a6db1c75991187c91093c1');
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.license).toBe('CC BY-NC-SA 4.0');
    expect(entry.crop_box_px).toBeUndefined();
    for(const [path,hash]of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot,entry.asset)))
      .toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    const svg=readFileSync(resolve(repositoryRoot,entry.asset),'utf8');
    expect(svg).toContain('viewBox="0 0 295.80533 173.81467"');
    expect(svg).not.toMatch(/(?:href|src)="https?:\/\//);
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/neural-lm-bengio-2026-11-03/')){
    const prefix='00 Учебник/Assets/Figures/curated/neural-lm-bengio-2026-11-03/';
    const [spec]=JSON.parse(readFileSync(resolve(repositoryRoot,prefix+'assets.json'),'utf8'));
    expect(entry).toEqual(spec);
    expect(entry.asset).toBe(prefix+'bengio-original-figure1.png');
    expect(entry.source_asset_sha256).toBe('0bddf9a608d62da449a6413d86cbf6e8b206212595efe0e0a14da0d79d5f4976');
    expect(entry.sha256).toBe('4027648579190044a162655b51453f816cdb98adb298d4c5a2a55bb0f6cdfcd1');
    expect(entry.license).toBe('CC BY (JMLR publication policy; version unspecified)');
    expect(entry.provenance_confirmation).toBe('publisher-publication-policy');
    expect(entry.derivation).toBe('source-pdf-render-crop');
    expect(entry.source_page).toBe(6);
    expect(entry.source_printed_page).toBe(1142);
    expect(entry.render_dpi).toBe(240);
    expect(entry.render_page_size_px).toEqual([2040,2640]);
    expect(entry.crop_box_px).toEqual([273,283,1780,1263]);
    for(const [p,h]of [[entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])expect(sha256(resolve(repositoryRoot,p!))).toBe(h);
    const policy=readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8');
    expect(policy).toMatch(/granting a CC-by\s+license to the public/);
    expect(readFileSync(resolve(repositoryRoot,spec.public_license_copy),'utf8')).toContain('No version is inferred here.');
    const bytes=execFileSync(process.env.PDFTOPPM??'pdftoppm',
      ['-f','6','-l','6','-singlefile','-r','240','-x','273','-y','283','-W','1507','-H','980',
        '-png',resolve(repositoryRoot,entry.source_asset!)],
      {env:{...process.env,LC_ALL:'C',LANG:'C'},maxBuffer:4*1024*1024});
    expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(bytes);
    expect(validateRenderedPngBytes(bytes)).toEqual([1507,980]);
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/ngram-makemore-2026-11-03/')){
    const prefix='00 Учебник/Assets/Figures/curated/ngram-makemore-2026-11-03/';
    const [spec]=JSON.parse(readFileSync(resolve(repositoryRoot,prefix+'assets.json'),'utf8'));
    expect(entry).toEqual(spec);
    expect(entry.asset).toBe(prefix+'karpathy-bigram-counts.png');
    expect(entry.derivation).toBe('notebook-output-extraction');
    expect(entry.license).toBe('MIT');
    expect(entry.commit).toBe('73c3fcc741f0ec104ca850b1fb0df90e7e8d4cde');
    expect(entry.source_asset_sha256).toBe('db2f139409bad2966baac06581b6ca8ec9a03668df50dd073577fd98b3be0045');
    expect(entry.sha256).toBe('0972425d7afc53f7d81c7da5aca31a4dd99cb664bf102f604b0e3c61cec57d39');
    for(const [path,digest]of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256],
      [spec.source_data,spec.source_data_sha256]]) expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(spec.source_cell_index).toBe(11);
    expect(spec.source_output_index).toBe(0);
    expect(entry.source_member).toBe('/cells/11/outputs/0/data/image~1png');
    const notebook=JSON.parse(readFileSync(resolve(repositoryRoot,entry.source_asset!),'utf8'));
    const encoded=notebook.cells[11].outputs[0].data['image/png'];
    const bytes=readFileSync(resolve(repositoryRoot,entry.asset));
    expect(bytes).toEqual(Buffer.from(Array.isArray(encoded)?encoded.join(''):encoded,'base64'));
    expect(validateRenderedPngBytes(bytes)).toEqual([884,884]);
    expect(readFileSync(resolve(repositoryRoot,spec.public_license_copy)))
      .toEqual(readFileSync(resolve(repositoryRoot,spec.license_evidence)));
    expect(spec.license_evidence_sha256).toBe('a59ec5cdb1c1e447e5266a014b3e7ded511f8d6e5a08931e931c87490ff821fe');
    expect(spec.source_data_url).toContain('/8f79bd0126eaef5eb82d346c9f85ae20ea1c476a/names.txt');
    expect(entry.crop_box_px).toBeUndefined();
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/normalization-course-2026-11-03/')){
    const prefix='00 Учебник/Assets/Figures/curated/normalization-course-2026-11-03/';
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,prefix+'assets.json'),'utf8'));
    const spec=specs.find((a:CuratedAsset)=>a.asset===entry.asset);
    expect(spec).toBeDefined();
    expect(entry).toEqual(spec);
    expect(entry.derivation).toBe('notebook-output-extraction');
    expect(entry.license).toBe('MIT');
    expect(entry.source_asset_sha256).toBe('464c959353b5e929bf0f922c0e7547823fb514fa2f672f30a065a9d403c641e4');
    expect(sha256(resolve(repositoryRoot,entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(entry.sha256);
    const notebook=JSON.parse(readFileSync(resolve(repositoryRoot,entry.source_asset!),'utf8'));
    expect([15,16]).toContain(spec.source_cell_index);
    expect(spec.source_output_index).toBe(2);
    const encoded=notebook.cells[spec.source_cell_index].outputs[spec.source_output_index].data['image/png'];
    expect(readFileSync(resolve(repositoryRoot,entry.asset)))
      .toEqual(Buffer.from(Array.isArray(encoded)?encoded.join(''):encoded,'base64'));
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(spec.dimensions_px);
    expect(spec.license_evidence_sha256).toBe('a59ec5cdb1c1e447e5266a014b3e7ded511f8d6e5a08931e931c87490ff821fe');
    expect(sha256(resolve(repositoryRoot,spec.license_evidence))).toBe(spec.license_evidence_sha256);
    expect(readFileSync(resolve(repositoryRoot,spec.public_license_copy)))
      .toEqual(readFileSync(resolve(repositoryRoot,spec.license_evidence)));
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/augmentation-composition-2026-11-03/')){
    const prefix='00 Учебник/Assets/Figures/curated/augmentation-composition-2026-11-03/';
    const [spec]=JSON.parse(readFileSync(resolve(repositoryRoot,prefix+'assets.json'),'utf8'));
    expect(entry.asset).toBe(prefix+'d2l-composed-transforms.svg');
    expect(entry).toEqual(spec);
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.license).toBe('CC BY-SA 4.0');
    expect(entry.source_image_url).toBe('https://d2l.ai/_images/output_image-augmentation_7d0887_81_0.svg');
    expect(entry.sha256).toBe('1ccc8d5cbf7f8dc340ca38e0dd2cf6aa9b915ec07c61d3d2a5be8dd7985e4c8f');
    for(const [pathKey,hashKey]of [
      ['source_asset','source_asset_sha256'],['source_page_snapshot','source_page_sha256'],
      ['source_code_snapshot','source_code_sha256'],['license_evidence','license_evidence_sha256']])
      expect(sha256(resolve(repositoryRoot,spec[pathKey]))).toBe(spec[hashKey]);
    const image=readFileSync(resolve(repositoryRoot,entry.asset));
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(entry.sha256);
    expect(image).toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    const svg=image.toString();
    expect(svg.match(/<image\b/g)).toHaveLength(8);
    expect(svg).not.toMatch(/(?:href|src)="https?:/);
    expect(svg).toContain('viewBox="0 0 465.600016 237.203471"');
    expect(readFileSync(resolve(repositoryRoot,spec.public_license_copy)))
      .toEqual(readFileSync(resolve(repositoryRoot,spec.license_evidence)));
    const html=readFileSync(resolve(repositoryRoot,spec.source_page_snapshot),'utf8');
    expect(html.split('id="pytorch-17-0"')[1].split('id="mxnet-17-1"')[0])
      .toContain('output_image-augmentation_7d0887_81_0.svg');
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/data-augmentation-2026-10-28/')) {
    const prefix='00 Учебник/Assets/Figures/curated/data-augmentation-2026-10-28/';
    const packet='05 Источники/Originals/data-augmentation-2026-10-28/';
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,prefix+'assets.json'),'utf8'));
    const name=entry.asset.slice(prefix.length);
    expect(['cutmix-comparison.png','d2l-random-crops.svg']).toContain(name);
    expect(entry).toEqual(specs.find((row:CuratedAsset)=>row.asset===entry.asset));
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    const bytes=readFileSync(resolve(repositoryRoot,entry.asset));
    expect(bytes).toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    expect(entry.sha256).toBe(entry.source_asset_sha256);
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.crop_box_px).toBeUndefined();
    if(name==='cutmix-comparison.png'){
      expect(entry.source_asset).toBe(packet+'cutmix/img1.PNG');
      expect(entry.commit).toBe('2d8eb68faff7fe4962776ad51d175c3b01a25734');
      expect(entry.sha256).toBe('7618a576a5ba78601fc8611c064d894b99c1572830dba5cfbd5424859dfb4c02');
      expect(entry.license).toBe('MIT');
      expect(entry.derivation).toBe('source-raster-copy');
      expect(entry.dimensions_px).toEqual([649,447]);
      expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual([649,447]);
      expectDecodableImage(resolve(repositoryRoot,entry.asset));
      expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8'))
        .toContain('software and associated documentation files');
    }else{
      expect(entry.source_asset).toBe(packet+'d2l/random-crops.svg');
      expect(entry.commit).toBe('23d7a5aecceee57d1292c56e90cce307f183bb0a');
      expect(entry.sha256).toBe('0aa186faaee2267fa0afe51adec4cf8daca445dfb71a292c236ff2aeb00294d3');
      expect(entry.license).toBe('CC BY-SA 4.0');
      expect(entry.derivation).toBe('source-svg-copy');
      const svg=bytes.toString('utf8');
      expect(svg.match(/viewBox="([^"]+)"/)?.[1].split(/\s+/).map(Number))
        .toEqual([0,0,465.600016,237.203471]);
      expect(svg.match(/<image\s/g)).toHaveLength(8);
      expect(svg).not.toMatch(/(?:href|src)="https?:\/\//);
      expect(readFileSync(resolve(repositoryRoot,entry.source_page_snapshot!),'utf8'))
        .toContain('output_image-augmentation_7d0887_45_0.svg');
    }
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/harvard-data-selection-2026-10-28/')) {
    const prefix='00 Учебник/Assets/Figures/curated/harvard-data-selection-2026-10-28/';
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,prefix+'assets.json'),'utf8'));
    expect(entry).toEqual(rows.find((row:CuratedAsset)=>row.asset===entry.asset));
    const specs:Record<string,{name:string;hash:string}>={
      'optimization-stack.svg':{name:'fig-optimization-stack.svg',hash:'c60797d3a177519453b8ec03a4c9c36f5e0fc99744c57ae96db11dcd97c24bb1'},
      'pretraining-amortization.png':{name:'fig-amortization-comparison.png',hash:'c93a906fb3551eaefcd763fad6501b48762637cb0bc20ed4d0e63d7da4e43566'},
    };
    const name=entry.asset.slice(prefix.length),spec=specs[name];
    expect(spec).toBeTruthy();
    expect(entry.source_asset).toBe('05 Источники/Originals/data-selection-reconciliation-2026-10-28/generated/'+spec.name);
    expect(entry.sha256).toBe(spec.hash);
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(spec.hash);
    expect(readFileSync(resolve(repositoryRoot,entry.asset)))
      .toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    expect(entry.license).toBe('CC BY-NC-ND 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-nc-nd/4.0/');
    expect(entry.source_page_sha256).toBe('9f31e8439bd840b3f6993851089acd305b230070bdd4fd6474c68fa39192f09e');
    expect(sha256(resolve(repositoryRoot,entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.rights_status).toBe('source-license-verified');
    if(name.endsWith('.svg')){
      expect(entry.derivation).toBe('source-svg-copy');
      expect(entry.source_viewbox).toEqual([0,0,686.95599,230.756]);
    }else{
      expect(entry.derivation).toBe('source-raster-copy');
      expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual([3658,2219]);
      expectDecodableImage(resolve(repositoryRoot,entry.asset));
    }
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/gan-dcgan-2026-10-28/')) {
    const prefix='00 Учебник/Assets/Figures/curated/gan-dcgan-2026-10-28/';
    const packet='05 Источники/Originals/gan-dcgan-2026-10-28/d2l/';
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,prefix+'assets.json'),'utf8'));
    expect(entry).toEqual(specs.find((row:CuratedAsset)=>row.asset===entry.asset));
    const hashes:Record<string,string>={
      'gan.svg':'f51dc373c89a89a620cef10e9712e991bd1c23dd4c62cbd698e56740caad1d5b',
      'gaussian-data.svg':'d572fe806140592e1a0f8af965df9ccda076bc75fd5e93c818f00178c407cadc',
      'gaussian-training.svg':'c1688624166f2eb43b85a70c3762fa6ac4f74cf73fb3e18224d5d5d8cef6b401',
      'leaky-relu.svg':'848a59ec192959be721f8a706c587f2adc8b79cc4c815a2d9d01e40210eccf31',
    };
    const name=entry.asset.slice(prefix.length);
    expect(entry.sha256).toBe(hashes[name]);
    expect(entry.source_asset).toBe(packet+name);
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.license).toBe('CC BY-SA 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-sa/4.0/');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.crop_box_px).toBeUndefined();
    expect(readFileSync(resolve(repositoryRoot,entry.asset)))
      .toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(hashes[name]);
    expect(sha256(resolve(repositoryRoot,entry.source_page_snapshot!)))
      .toBe(entry.source_page_sha256);
    expect(sha256(resolve(repositoryRoot,entry.license_evidence!)))
      .toBe(entry.license_evidence_sha256);
    const svg=readFileSync(resolve(repositoryRoot,entry.asset),'utf8');
    expect(svg.match(/viewBox="([^"]+)"/)?.[1].trim().split(/\s+/).map(Number))
      .toEqual(entry.source_viewbox);
    expect(svg).not.toMatch(/(?:href|src)="https?:\/\//);
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/selection-engineering-2026-10-28/')) {
    const prefix='00 Учебник/Assets/Figures/curated/selection-engineering-2026-10-28/';
    const packet='05 Источники/Originals/selection-engineering-2026-10-28/';
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,prefix+'assets.json'),'utf8'));
    const spec=specs.find((row:CuratedAsset)=>row.asset===entry.asset);
    expect(spec).toBeTruthy();
    for(const [key,value] of Object.entries(spec))
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    const name=entry.asset.slice(prefix.length);
    expect(['selection-inequality.png','performance-data.png','distributed-selection.svg','selection-decision-tree.svg']).toContain(name);
    expect(entry.source_asset).toBe(packet+name);
    expect(entry.source_page_snapshot).toBe(packet+'web-2026-10-28.html');
    expect(entry.source_page_sha256).toBe('9f31e8439bd840b3f6993851089acd305b230070bdd4fd6474c68fa39192f09e');
    expect(sha256(resolve(repositoryRoot,entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    const html=readFileSync(resolve(repositoryRoot,entry.source_page_snapshot!),'utf8');
    expect(html).toContain('https://creativecommons.org/licenses/by-nc-nd/4.0/');
    expect(html).toContain(entry.source_image_url!.split('/').at(-1));
    expect(entry.license).toBe('CC BY-NC-ND 4.0');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-nc-nd/4.0/');
    expect(entry.source_url).toMatch(/^https:\/\/mlsysbook.ai\/vol1\/09_data_selection\/09_data_selection.html#fig-/);
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.render_dpi).toBeUndefined();
    expect(readFileSync(resolve(repositoryRoot,entry.asset)))
      .toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(entry.sha256);
    expect(entry.sha256).toBe(entry.source_asset_sha256);
    if(name.endsWith('.png')){
      expect(entry.derivation).toBe('source-raster-copy');
      expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(entry.dimensions_px);
      expectDecodableImage(resolve(repositoryRoot,entry.asset));
    } else {
      expect(entry.derivation).toBe('source-svg-copy');
      const viewBox=readFileSync(resolve(repositoryRoot,entry.asset),'utf8')
        .match(/viewBox="([^"]+)"/)?.[1].trim().split(/\s+/).map(Number);
      expect(viewBox).toEqual(name==='selection-decision-tree.svg'
        ? [0,0,761.21869,352.48932] : [0,0,354.23468,143.04933]);
    }
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/mae-2026-10-28/')) {
    const prefix='00 Учебник/Assets/Figures/curated/mae-2026-10-28/';
    const packet='05 Источники/Originals/mae-2026-10-28/';
    const pin='efb2a8062c206524e35e47d04501ed4f544c0ae8';
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,prefix+'assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    const specs:Record<string,{source:string;hash:string;cell?:number;size:number[]}>= {
      'architecture.png':{source:'architecture-release.png',hash:'468cb0ed6bcd501bf3c05e257d253fc7dbd6e36efedbca8e7372bdcffaa80467',size:[2022,1158]},
      'pixel-reconstruction.png':{source:'mae/mae_visualize.ipynb',hash:'66052e4579206d1b8b277bf813ac0458f835b94dcf78772f28b3c7edf932fd0b',cell:10,size:[1353,323]},
      'gan-comparison.png':{source:'mae/mae_visualize.ipynb',hash:'66052e4579206d1b8b277bf813ac0458f835b94dcf78772f28b3c7edf932fd0b',cell:14,size:[1353,323]},
    };
    const spec=specs[entry.asset.slice(prefix.length)];
    expect(spec).toBeDefined();
    expect(entry.commit).toBe(pin);
    expect(entry.source_asset).toBe(packet+spec.source);
    expect(entry.source_asset_sha256).toBe(spec.hash);
    expect(entry.source_page_snapshot).toBe(packet+'mae/README.md');
    expect(entry.source_page_sha256).toBe('edda3ecd28d6cb32559e638c56c7d28d59d6e2f9ada2dd084292c8706764bea6');
    expect(entry.license).toBe('CC BY-NC 4.0');
    expect(entry.license_evidence).toBe(packet+'mae/LICENSE');
    expect(entry.license_evidence_sha256).toBe('ffe3b1e0b58c1f1250ce48f84d8c37328f9a1ff59bb77b63bc97480cf2778b64');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.source_cell).toBe(spec.cell);
    expect(entry.dimensions_px).toEqual(spec.size);
    for(const [path,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    const sourceBytes=readFileSync(resolve(repositoryRoot,entry.source_asset!));
    let expected=sourceBytes;
    if(spec.cell!==undefined){
      expect(entry.derivation).toBe('notebook-output-extraction');
      const notebook=JSON.parse(sourceBytes.toString('utf8'));
      const output=notebook.cells[spec.cell].outputs.find((x:any)=>x.data?.['image/png']);
      const encoded=output.data['image/png'];
      expected=Buffer.from(Array.isArray(encoded)?encoded.join(''):encoded,'base64');
    }else expect(entry.derivation).toBe('source-raster-copy');
    expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(expected);
    expect(validateRenderedPngBytes(expected)).toEqual(spec.size);
    return;
  }
  if(entry.asset==='00 Учебник/Assets/Figures/curated/moco-2026-10-28/moco-dictionary.png') {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/moco-2026-10-28/assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows[0]);
    expect(entry.license).toBe('CC BY-NC 4.0');
    expect(entry.source_asset_sha256).toBe('ad31ea26ed2d9ba3e2f767a2dcb0961a5b70ff505b1c3fb18d07201140e4e1fe');
    expect(entry.permission_asset_sha256).toBe('720c29e985b87fbd10e55b854365587ef232467c20203f7aedd3109f666961bd');
    expect(entry.source_page).toBe(1);
    expect(entry.render_dpi).toBe(400);
    expect(entry.crop_box_px).toEqual([1940,1190,2840,1900]);
    expect(entry.dimensions_px).toEqual([900,710]);
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256],
      [entry.permission_asset,entry.permission_asset_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    const rendered=execFileSync('pdftoppm',['-f','1','-l','1','-r','400',
      '-x','1940','-y','1190','-W','900','-H','710','-singlefile','-png',
      resolve(repositoryRoot,entry.source_asset!)],
      {env:{...process.env,LC_ALL:'C',LANG:'C'},maxBuffer:4*1024*1024});
    expect(rendered).toEqual(readFileSync(resolve(repositoryRoot,entry.asset)));
    expect(validateRenderedPngBytes(rendered)).toEqual([900,710]);
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/simclr-2026-10-28/')) {
    const specs:Record<string,[number,number[]]>={
      'framework.png':[2,[1735,350,2985,1180]],
      'crop-views.png':[3,[1735,350,2985,940]],
      'augmentations.png':[4,[320,350,3010,1485]],
      'projection-head.png':[6,[440,1035,1440,1610]],
    };
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/simclr-2026-10-28/assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    const name=entry.asset.slice(entry.asset.lastIndexOf('/')+1);
    expect(Object.keys(specs)).toContain(name);
    const [page,box]=specs[name]!;
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.source_asset_sha256).toBe('cc620e511bafe8b4b11ca3415f6d7bcf127a05818087bfa49905de3a2e50bc2d');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.source_page).toBe(page);
    expect(entry.render_dpi).toBe(400);
    expect(entry.crop_box_px).toEqual(box);
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8').replace(/\s+/g,' '))
      .toContain('Creative Commons Attribution 4.0 International License');
    const rendered=execFileSync('pdftoppm',['-f',String(page),'-l',String(page),'-r','400',
      '-x',String(box[0]),'-y',String(box[1]),'-W',String(box[2]!-box[0]!),
      '-H',String(box[3]!-box[1]!),'-singlefile','-png',resolve(repositoryRoot,entry.source_asset!)],
      {env:{...process.env,LC_ALL:'C',LANG:'C'},maxBuffer:8*1024*1024});
    expect(rendered).toEqual(readFileSync(resolve(repositoryRoot,entry.asset)));
    expect(validateRenderedPngBytes(rendered)).toEqual(entry.dimensions_px);
    return;
  }
  if(entry.asset==='00 Учебник/Assets/Figures/curated/semi-supervised-2026-10-28/fixmatch-diagram.png') {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/semi-supervised-2026-10-28/assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    expect(entry.license).toBe('Apache-2.0');
    expect(entry.source_page).toBe(2);
    expect(entry.render_dpi).toBe(400);
    expect(entry.crop_box_px).toEqual([800,375,2650,1075]);
    expect(entry.dimensions_px).toEqual([1850,700]);
    expect(entry.derivation).toBe('pdf-page-render-crop');
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256],
      [entry.permission_asset,entry.permission_asset_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(entry.permission_asset_sha256).toBe('fdd7acf0fa895003968e1c747990801301fe263530b5a58a03979a927bfe6e6f');
    const rendered=execFileSync('pdftoppm',['-f','2','-l','2','-r','400',
      '-x','800','-y','375','-W','1850','-H','700','-singlefile','-png',
      resolve(repositoryRoot,entry.source_asset!)],
      {env:{...process.env,LC_ALL:'C',LANG:'C'},maxBuffer:6*1024*1024});
    expect(rendered).toEqual(readFileSync(resolve(repositoryRoot,entry.asset)));
    expect(validateRenderedPngBytes(rendered)).toEqual([1850,700]);
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/geometric-coresets-2026-10-28/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/geometric-coresets-2026-10-28/assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    expect(entry.license).toBe('CC BY-NC-SA 4.0');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.source_page).toBe(14);
    expect(entry.render_dpi).toBe(288);
    expect(entry.crop_box_px).toEqual([620,1452,2120,2308]);
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    const permissions=readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8');
    expect(permissions).toContain('All other lecture notes');
    expect(permissions).toContain('Attribution-NonCommercial-ShareAlike 4.0');
    const rerendered=execFileSync('pdftoppm',['-f','14','-l','14','-r','288',
      '-x','620','-y','1452','-W','1500','-H','856','-singlefile','-png',
      resolve(repositoryRoot,entry.source_asset!)],
      {env:{...process.env,LC_ALL:'C',LANG:'C'},maxBuffer:4*1024*1024});
    expect(rerendered).toEqual(readFileSync(resolve(repositoryRoot,entry.asset)));
    expect(validateRenderedPngBytes(rerendered)).toEqual([1500,856]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/autodiff-2026-10-27/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/autodiff-2026-10-27/assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    expect(entry.source_asset).toBe('05 Источники/Originals/frameworks-2026-10-27/baydin-jmlr-17-468.pdf');
    expect(entry.source_asset_sha256).toBe('93eb2f7025ea098c18a65b428ac991dfacec2de56618c4e25c6a22b9cf443c59');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.render_dpi).toBe(400);
    expect(entry.render_page_size_px).toEqual([3400,4400]);
    expect([9,10,13]).toContain(entry.source_page);
    expect(entry.source_url).toBe('https://jmlr.org/papers/volume18/17-468/17-468.pdf#page='+entry.source_page);
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8'))
      .toContain('License: CC-BY 4.0');
    const box=entry.crop_box_px as number[];
    expect(box).toHaveLength(4);
    expect(box[0]).toBeGreaterThanOrEqual(0);
    expect(box[1]).toBeGreaterThanOrEqual(0);
    expect(box[2]).toBeLessThanOrEqual(3400);
    expect(box[3]).toBeLessThanOrEqual(4400);
    expect(entry.dimensions_px).toEqual([box[2]-box[0],box[3]-box[1]]);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual(entry.dimensions_px);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/mbu-review-2026-10/')) {
    const archive = '05 Источники/Originals/mbu-2026-10-26';
    const refs = JSON.parse(readFileSync(resolve(repositoryRoot, archive, 'references.json'), 'utf8'));
    const original = refs.figures.find((row: {file: string}) => entry.asset.endsWith('/' + row.file));
    expect(original).toBeTruthy();
    expect(entry.source_image_url).toBe(original.url);
    expect(entry.sha256).toBe(original.sha256);
    expect(entry.source_asset).toBe(entry.asset);
    expect(entry.source_asset_sha256).toBe(original.sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(original.sha256);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot, entry.asset))))
      .toEqual(original.dimensions);
    expect(entry.dimensions_px).toEqual(original.dimensions);
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.license).toBe('MIT');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    for (const [path, hash] of [[entry.source_page_snapshot, entry.source_page_sha256],
                               [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.public_license_copy!)))
      .toEqual(readFileSync(resolve(repositoryRoot, archive, 'jax-LICENSE-MIT.txt')));
    return;
  }
  const wengLegacyHashes: Record<string,string> = {
    'curated-weng-ae-00000000009b':'07726792bd2f1cc0dd0400720cc5149ff51c4845bf8ccd4ee4cb130ac8fc7eb4',
    'curated-weng-vae-00000000009c':'5d668275177fac4b1ec308a83a6e575108f543749249bcea8a6c2992361c8cba',
    'curated-weng-reparam-00000000009d':'b25dedb6fafa3e9b1344b315b0853db99504754006e52feda2539e8873e12d9b'
  };
  if (entry.id && wengLegacyHashes[entry.id]) {
    expect(entry.provenance_confirmation).toBe('legacy-attribution-reviewed-permission-unconfirmed');
    expect(entry.metadata_status).toBe('needs-confirmation');
    expect(entry.rights_status).toBe('permission-unconfirmed');
    expect(entry.license).toBe('Historical reproduction record; author permission not independently confirmed');
    expect(entry.sha256).toBe(wengLegacyHashes[entry.id]);
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot,entry.asset)))
      .toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/winograd-2026-10-27/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/winograd-2026-10-27/assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    expect(entry.derivation).toBe('source-pdf-render');
    expect(entry.render_dpi).toBe(400);
    expect(entry.source_page).toBe(4);
    expect(entry.crop_box_px).toEqual([572,305,1867,1583]);
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual([1295,1278]);
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/mig-2026-10-26/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/mig-2026-10-26/assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    expect(entry.derivation).toBe('source-pdf-render');
    expect(entry.render_dpi).toBe(400);
    expect(entry.crop_box_px).toBeUndefined();
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual(entry.dimensions_px);
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/finops-2026-10-26/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/finops-2026-10-26/assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    expect(entry.derivation).toBe('source-code-render');
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_code,entry.source_code_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    const source=readFileSync(resolve(repositoryRoot,entry.source_code!),'utf8');
    const section=source.split('::: {#fig-tco-iceberg ')[1].split('\n:::')[0];
    const cell=section.match(/^```\{python\}\n([\s\S]*?)^```/m)![1];
    expect(createHash('sha256').update(cell).digest('hex')).toBe(entry.source_member_sha256);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual(entry.dimensions_px);
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/architecture-lab-2026-10-26/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/architecture-lab-2026-10-26/assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_code,entry.source_code_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sourceSnapshotSha256(repositoryRoot,path!)).toBe(hash);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    expect(entry.crop_box_px).toBeUndefined();
    if(entry.source_asset) {
      expect(entry.derivation).toBe('source-svg-copy');
      expect(readFileSync(resolve(repositoryRoot,entry.asset)))
        .toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset)));
    } else {
      expect(entry.derivation).toBe('source-code-render');
      const report=JSON.parse(execFileSync('python3',
        ['publishing/tools/check_architecture_lab.py'],
        {cwd:repositoryRoot,encoding:'utf8',env:{...process.env,LC_ALL:'C',LANG:'C'}}));
      expect(entry.source_member_sha256).toBe(report.source_plot_source_sha256);
      expect(report.source_candidates).toBe(12);
    }
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/semi-supervised-extensions-2026-10-29/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/semi-supervised-extensions-2026-10-29/assets.json'),'utf8'));
    const row=rows.find((r:CuratedAsset)=>r.asset===entry.asset);
    expect(entry).toEqual(row);
    for(const [path,hash] of [[row.asset,row.sha256],
      [row.source_asset,row.source_asset_sha256],
      [row.license_evidence,row.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path))).toBe(hash);
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.render_dpi).toBe(240);
    expect(entry.source_page).toBe(entry.asset.endsWith('tmix-original.png')?1:5);
    expect(row.crop_box_points).toEqual(entry.source_page===1
      ?[303,220,532,475]:[68,68,532,278]);
    expect(entry.license).toBe('CC BY 4.0');
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual(entry.dimensions_px);
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8'))
      .toContain('As of 2016');
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/consistency-training-2026-10-29/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/consistency-training-2026-10-29/assets.json'),'utf8'));
    const row=rows.find((r:CuratedAsset)=>r.asset===entry.asset);
    expect(entry).toEqual(row);
    for(const [path,hash] of [[row.asset,row.sha256],
      [row.source_asset,row.source_asset_sha256],
      [row.license_evidence,row.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path))).toBe(hash);
    if(entry.asset.endsWith('.gif')) {
      expect(entry.derivation).toBe('source-gif-copy');
      expect(entry.commit).toBe('c5125d267531ce0f10b2238cf95604d287de63c8');
      expect(entry.frames).toBe(14);
      expect(entry.dimensions_px).toEqual([320,180]);
      expect(entry.license).toBe('MIT');
      expect(readFileSync(resolve(repositoryRoot,entry.asset)))
        .toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset!)));
      expectDecodableImage(resolve(repositoryRoot,entry.asset));
    } else {
      expect(entry.derivation).toBe('pdf-page-render-crop');
      expect(entry.render_dpi).toBe(400);
      expect(entry.source_page).toBe(entry.asset.endsWith('uda-fixmatch.png')?5:4);
      expect(entry.license).toBe('CC BY 4.0');
      expect(row.crop_box_points).toEqual(entry.asset.endsWith('uda-fixmatch.png')
        ?[68,65,529,253]:[70,64,293,260]);
      expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
        .toEqual(entry.dimensions_px);
      const policy=readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8');
      expect(policy).toContain('Creative Commons 4.0 BY');
      expect(policy).toContain('As of 2016');
    }
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/autoencoder-course-2026-11-03/')){
    const specs=[{"id":"curated-uva-autoencoder-output-472ec692b4af","asset":"00 Учебник/Assets/Figures/curated/autoencoder-course-2026-11-03/uva-latent-width.svg","author":"Phillip Lippe","title":"Validation reconstruction error over latent dimensionality","source_url":"https://github.com/phlippe/uvadlc_notebooks/blob/c7d9e18f42fee53bc8925f08ec2358371b1e9200/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb","source_image_url":"https://raw.githubusercontent.com/phlippe/uvadlc_notebooks/c7d9e18f42fee53bc8925f08ec2358371b1e9200/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb","source_asset":"05 Источники/Originals/autoencoder-2026-10-26/uva-ae.ipynb","source_asset_sha256":"ae9cb1d0c65d447744eadca5aeccec52597b9e9191692d7873344e215439a6be","source_page_snapshot":"05 Источники/Originals/autoencoder-2026-10-26/uva-ae.ipynb","source_page_sha256":"ae9cb1d0c65d447744eadca5aeccec52597b9e9191692d7873344e215439a6be","source_cell":26,"source_member":"cells/26/outputs/0/data/image/svg+xml","source_member_sha256":"472ec692b4afb5b51012b78d61ac8cd8ae827303b105fb7cd3a3172ad188bee3","commit":"c7d9e18f42fee53bc8925f08ec2358371b1e9200","license":"MIT","license_url":"https://github.com/phlippe/uvadlc_notebooks/blob/c7d9e18f42fee53bc8925f08ec2358371b1e9200/LICENSE.md","license_evidence":"site/public/licenses/uva-c7d9e18f-MIT.txt","license_evidence_sha256":"bc760ea73079016ab58ddb9ade0ea6e822dc1ba801b8e26c9d0eadb68e564735","derivation":"notebook-svg-output-copy","modifications":"Complete original SVG output copied byte-for-byte, including any embedded raster. No plotting, crop, redraw, relabeling, resizing or recompression.","metadata_status":"verified","provenance_confirmation":"source-license-verified","last_verified":"2026-11-03","sha256":"472ec692b4afb5b51012b78d61ac8cd8ae827303b105fb7cd3a3172ad188bee3","bytes":31342,"viewBox":[0,0,392.025469,282.535781],"used_in":["00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE.md"]},{"id":"curated-uva-autoencoder-output-5fefe3cd0a33","asset":"00 Учебник/Assets/Figures/curated/autoencoder-course-2026-11-03/uva-noise-reconstruction.svg","author":"Phillip Lippe","title":"Reconstruction of independently generated noise","source_url":"https://github.com/phlippe/uvadlc_notebooks/blob/c7d9e18f42fee53bc8925f08ec2358371b1e9200/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb","source_image_url":"https://raw.githubusercontent.com/phlippe/uvadlc_notebooks/c7d9e18f42fee53bc8925f08ec2358371b1e9200/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb","source_asset":"05 Источники/Originals/autoencoder-2026-10-26/uva-ae.ipynb","source_asset_sha256":"ae9cb1d0c65d447744eadca5aeccec52597b9e9191692d7873344e215439a6be","source_page_snapshot":"05 Источники/Originals/autoencoder-2026-10-26/uva-ae.ipynb","source_page_sha256":"ae9cb1d0c65d447744eadca5aeccec52597b9e9191692d7873344e215439a6be","source_cell":32,"source_member":"cells/32/outputs/0/data/image/svg+xml","source_member_sha256":"5fefe3cd0a33955b020f7343acd2d3da9dffa004df4f773f1c657d0ac089cc13","commit":"c7d9e18f42fee53bc8925f08ec2358371b1e9200","license":"MIT","license_url":"https://github.com/phlippe/uvadlc_notebooks/blob/c7d9e18f42fee53bc8925f08ec2358371b1e9200/LICENSE.md","license_evidence":"site/public/licenses/uva-c7d9e18f-MIT.txt","license_evidence_sha256":"bc760ea73079016ab58ddb9ade0ea6e822dc1ba801b8e26c9d0eadb68e564735","derivation":"notebook-svg-output-copy","modifications":"Complete original SVG output copied byte-for-byte, including any embedded raster. No plotting, crop, redraw, relabeling, resizing or recompression.","metadata_status":"verified","provenance_confirmation":"source-license-verified","last_verified":"2026-11-03","sha256":"5fefe3cd0a33955b020f7343acd2d3da9dffa004df4f773f1c657d0ac089cc13","bytes":119845,"viewBox":[0,0,405,130.885027],"used_in":["00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE.md"]},{"id":"curated-uva-autoencoder-output-f0f5ef6a8dec","asset":"00 Учебник/Assets/Figures/curated/autoencoder-course-2026-11-03/uva-pattern-reconstruction.svg","author":"Phillip Lippe","title":"Reconstruction of synthetic fields, blocks and ramps","source_url":"https://github.com/phlippe/uvadlc_notebooks/blob/c7d9e18f42fee53bc8925f08ec2358371b1e9200/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb","source_image_url":"https://raw.githubusercontent.com/phlippe/uvadlc_notebooks/c7d9e18f42fee53bc8925f08ec2358371b1e9200/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb","source_asset":"05 Источники/Originals/autoencoder-2026-10-26/uva-ae.ipynb","source_asset_sha256":"ae9cb1d0c65d447744eadca5aeccec52597b9e9191692d7873344e215439a6be","source_page_snapshot":"05 Источники/Originals/autoencoder-2026-10-26/uva-ae.ipynb","source_page_sha256":"ae9cb1d0c65d447744eadca5aeccec52597b9e9191692d7873344e215439a6be","source_cell":34,"source_member":"cells/34/outputs/0/data/image/svg+xml","source_member_sha256":"f0f5ef6a8decb6a29fd2975f70cd5d7918d0b483431486cd1ccba4140a06d63e","commit":"c7d9e18f42fee53bc8925f08ec2358371b1e9200","license":"MIT","license_url":"https://github.com/phlippe/uvadlc_notebooks/blob/c7d9e18f42fee53bc8925f08ec2358371b1e9200/LICENSE.md","license_evidence":"site/public/licenses/uva-c7d9e18f-MIT.txt","license_evidence_sha256":"bc760ea73079016ab58ddb9ade0ea6e822dc1ba801b8e26c9d0eadb68e564735","derivation":"notebook-svg-output-copy","modifications":"Complete original SVG output copied byte-for-byte, including any embedded raster. No plotting, crop, redraw, relabeling, resizing or recompression.","metadata_status":"verified","provenance_confirmation":"source-license-verified","last_verified":"2026-11-03","sha256":"f0f5ef6a8decb6a29fd2975f70cd5d7918d0b483431486cd1ccba4140a06d63e","bytes":58332,"viewBox":[0,0,405,227.11981],"used_in":["00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE.md"]},{"id":"curated-uva-autoencoder-output-94bd5c7d9898","asset":"00 Учебник/Assets/Figures/curated/autoencoder-course-2026-11-03/uva-random-codes.svg","author":"Phillip Lippe","title":"Ordinary decoder outputs from standard-normal codes","source_url":"https://github.com/phlippe/uvadlc_notebooks/blob/c7d9e18f42fee53bc8925f08ec2358371b1e9200/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb","source_image_url":"https://raw.githubusercontent.com/phlippe/uvadlc_notebooks/c7d9e18f42fee53bc8925f08ec2358371b1e9200/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb","source_asset":"05 Источники/Originals/autoencoder-2026-10-26/uva-ae.ipynb","source_asset_sha256":"ae9cb1d0c65d447744eadca5aeccec52597b9e9191692d7873344e215439a6be","source_page_snapshot":"05 Источники/Originals/autoencoder-2026-10-26/uva-ae.ipynb","source_page_sha256":"ae9cb1d0c65d447744eadca5aeccec52597b9e9191692d7873344e215439a6be","source_cell":37,"source_member":"cells/37/outputs/0/data/image/svg+xml","source_member_sha256":"94bd5c7d9898dbc7483f254e77a2b7d032e7efbe995a0cac7803aeefa1968722","commit":"c7d9e18f42fee53bc8925f08ec2358371b1e9200","license":"MIT","license_url":"https://github.com/phlippe/uvadlc_notebooks/blob/c7d9e18f42fee53bc8925f08ec2358371b1e9200/LICENSE.md","license_evidence":"site/public/licenses/uva-c7d9e18f-MIT.txt","license_evidence_sha256":"bc760ea73079016ab58ddb9ade0ea6e822dc1ba801b8e26c9d0eadb68e564735","derivation":"notebook-svg-output-copy","modifications":"Complete original SVG output copied byte-for-byte, including any embedded raster. No plotting, crop, redraw, relabeling, resizing or recompression.","metadata_status":"verified","provenance_confirmation":"source-license-verified","last_verified":"2026-11-03","sha256":"94bd5c7d9898dbc7483f254e77a2b7d032e7efbe995a0cac7803aeefa1968722","bytes":43514,"viewBox":[0,0,460.8,240.834783],"used_in":["00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE.md"]}];
    const spec=specs.find(x=>x.asset===entry.asset);
    expect(spec,entry.asset).toBeTruthy();
    expect(entry).toEqual(spec);
    for(const[path,hash]of [[entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    const notebook=JSON.parse(readFileSync(resolve(repositoryRoot,entry.source_asset!),'utf8'));
    const data=notebook.cells[entry.source_cell!].outputs[0].data['image/svg+xml'];
    const original=Buffer.from(Array.isArray(data)?data.join(''):data,'utf8');
    const published=readFileSync(resolve(repositoryRoot,entry.asset));
    expect(published).toEqual(original);
    expect(createHash('sha256').update(published).digest('hex')).toBe(entry.sha256);
    const svg=published.toString();
    expect(svg).toContain('viewBox="'+spec!.viewBox.join(' ')+'"');
    expect(svg).not.toContain('<script');
    expect(svg).not.toMatch(/<image[^>]+(?:href|xlink:href)="https?:/);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  const evaluationArchitecture = entry.asset === '00 Учебник/Assets/Figures/curated/linux-foundation-recommenders-architecture.png';
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/hardware-completion-2026-10-29/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/serving-operations-2026-10-29/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/recommender-data-2026-10-25/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/recommender-evaluation-2026-10-25/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/dropout-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/cnn-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/group-equivariance-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/rnn-execution-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/dlrm-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/reduction-primitives-2026-10-27/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/indexed-movement-2026-10-27/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/collective-destinations-2026-10-27/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/nn-native-2026-10-27/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/frameworks-native-2026-10-27/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/autoencoder-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/edls-neural-kv-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/http-lifecycle-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/deployment-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/workflow-feedback-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/model-export-2026-10-27/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/tensorflow-export-2026-10-27/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/jax-runtime-2026-10-27/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/tf-data-2026-10-28/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/galore-2026-10-28/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/data-selection-2026-10-28/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/active-learning-2026-10-28/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/semi-supervised-2026-10-28/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/feature-stores-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/platform-resources-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/planseqlearn-2026-10-26/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/recommender-mf-2026-10-25/') || evaluationArchitecture) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      evaluationArchitecture ? '00 Учебник/Assets/Figures/curated/recommender-evaluation-2026-10-25/assets.json'
      : entry.asset.substring(0,entry.asset.lastIndexOf('/'))+'/assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256]])
      expect(sourceSnapshotSha256(repositoryRoot,path!)).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot,entry.asset))
      .equals(readFileSync(resolve(repositoryRoot,entry.source_asset!)))).toBe(true);
    expect(entry.crop_box_px).toBeUndefined();
    if(entry.asset.endsWith('.png')) {
      expect(entry.derivation).toBe('source-raster-copy');
      expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
        .toEqual(entry.dimensions_px);
    } else if(entry.asset.endsWith('.gif')) {
      expect(entry.derivation).toBe('source-gif-copy');
      expectDecodableImage(resolve(repositoryRoot,entry.asset));
      const bytes=readFileSync(resolve(repositoryRoot,entry.asset));
      expect([bytes.readUInt16LE(6),bytes.readUInt16LE(8)]).toEqual(entry.dimensions_px);
    } else {
      expect(entry.derivation).toBe('source-svg-copy');
      expectDecodableImage(resolve(repositoryRoot,entry.asset));
    }
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/kernels-2026-11-03/')) {
    const folder = '00 Учебник/Assets/Figures/curated/kernels-2026-11-03/';
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,folder+'assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    const pins = {
      'sklearn-rbf-decision-grid.png':'32d1c64d0053c29dce038e18e28b6b203617bc6c6eabebd3ca2a8965556ba33b',
      'sklearn-rbf-validation-grid.png':'4ce8860a12309b462ae25ca56601c68daf74bbf8533d5648ee8323261cb5009a',
    } as Record<string,string>;
    const name = entry.asset.slice(folder.length);
    expect(pins[name]).toBeDefined();
    expect(entry.sha256).toBe(pins[name]);
    expect(entry.license).toBe('BSD-3-Clause');
    expect(entry.commit).toBe('199d116ab37beed506c63fc9f6a74f56e228de43');
    expect(entry.derivation).toBe('source-image-copy');
    expect(entry.source_asset).toBe(entry.asset);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.source_url).toBe('https://scikit-learn.org/1.9/auto_examples/svm/plot_rbf_parameters.html');
    const suffix=name==='sklearn-rbf-decision-grid.png'?'001':'002';
    expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/199d116ab37beed506c63fc9f6a74f56e228de43/1.9/_images/sphx_glr_plot_rbf_parameters_'+suffix+'.png');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.source_page_sha256).toBe('396ef426d23724920dbe4ebc477493729c882e49a5cfbb3cf6f90d13a2144bec');
    expect(entry.license_evidence_sha256).toBe('50d6a9d340f19ab355609917993114daf5f47e3161067bcf34955bbd05cd9cb0');
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot,entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    expect(sha256(resolve(repositoryRoot,entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset)))).toEqual([800,600]);
    expect(entry.dimensions_px).toEqual([800,600]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/gmm-2026-11-03/')) {
    const folder = '00 Учебник/Assets/Figures/curated/gmm-2026-11-03/';
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,folder+'assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    const pins = {"jake-digit-aic.png":"4cd0f062617e44b4312d4f58413c58337b8da8649984483591e84e2eb84df6e7","jake-generated-digits.png":"75ce77d9c4f701c68c1dd1a6d6ed33977a0e692e1e7e2fcfe400db2167a3b9ba","jake-moons-generated.png":"32d53b5e501feb04ad7b0eee235dc73443a03a3c060cfcdf493ca363144e6806","jake-training-digits.png":"7a0112379a3d2ed97ce6e09b011520c9fd7eba630000a4662640bd391c4295a6","sklearn-gmm-initialization.png":"ab5a8bd8e440a80fd83b0ce5ef7614c2d3362377fef0274176ac910951bb3707"} as Record<string,string>;
    const name = entry.asset.slice(folder.length);
    expect(entry.sha256).toBe(pins[name]);
    expect(pins[name]).toBeDefined();
    expect(entry.crop_box_px).toBeUndefined();
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(entry.sha256);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset)))).toEqual(entry.dimensions_px);
    expect(sha256(resolve(repositoryRoot,entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot,entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    if (name.startsWith('jake-')) {
      expect(entry.license).toBe('CC BY-NC-ND 3.0 Unported');
      expect(entry.commit).toBe('d66231454ef753818dc9213c9b5942e067266966');
      expect(entry.source_asset_sha256).toBe('9438d483e156738ba8d581e54f150590061ff6fbe6147cb38ab56ae0449c07e6');
      expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/d66231454ef753818dc9213c9b5942e067266966/notebooks/05.12-Gaussian-Mixtures.ipynb');
      const cells = {'jake-moons-generated.png':34,'jake-training-digits.png':42,
        'jake-digit-aic.png':46,'jake-generated-digits.png':52} as Record<string,number>;
      expect(entry.source_member).toBe('/cells/'+cells[name]+'/outputs/0/data/image~1png');
      const notebook = JSON.parse(readFileSync(resolve(repositoryRoot,entry.source_asset!),'utf8'));
      const data = notebook.cells[cells[name]].outputs[0].data['image/png'];
      expect(readFileSync(resolve(repositoryRoot,entry.asset)))
        .toEqual(Buffer.from(Array.isArray(data)?data.join(''):data,'base64'));
    } else {
      expect(name).toBe('sklearn-gmm-initialization.png');
      expect(entry.license).toBe('BSD-3-Clause');
      expect(entry.commit).toBe('199d116ab37beed506c63fc9f6a74f56e228de43');
      expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/199d116ab37beed506c63fc9f6a74f56e228de43/1.9/_images/sphx_glr_plot_gmm_init_001.png');
      expect(entry.source_asset_sha256).toBe(entry.sha256);
      expect(entry.source_page_sha256).toBe('82c5434602c464e1bdfac7483eed0b6d48cf1a41cd18f982182989f74b158731');
      expect(sha256(resolve(repositoryRoot,entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    }
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/gmm-2026-10-25/') ||
      entry.asset.startsWith('00 Учебник/Assets/Figures/curated/kernels-svm-rff-2026-10-25/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      entry.asset.substring(0,entry.asset.lastIndexOf('/'))+'/assets.json'),'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row=>row.asset===entry.asset));
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256]])
      if(path) expect(sha256(resolve(repositoryRoot,path))).toBe(hash);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual(entry.dimensions_px);
    if(entry.derivation==='notebook-output-extraction') {
      const match=entry.source_member?.match(/^\/cells\/(\d+)\/outputs\/0\/data\/image~1png$/);
      expect(match).not.toBeNull();
      const notebook=JSON.parse(readFileSync(resolve(repositoryRoot,entry.source_asset!),'utf8'));
      const data=notebook.cells[Number(match![1])].outputs[0].data['image/png'];
      const original=Buffer.from(Array.isArray(data)?data.join(''):data,'base64');
      expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(original);
    } else expect(entry.source_asset_sha256).toBe(entry.sha256);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/dimensionality-2026-11-03/')) {
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/dimensionality-2026-11-03/assets.json'), 'utf8')) as CuratedAsset[];
    const expected = rows.find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    expect(entry).toEqual(expected);
    const specifications = {
      "distill-global-150-input.png": {
        "sha256": "b4350204c7d3d88f96a01afe78eccb18d8773e3d10d1e35d5378f024cc6a8e48",
        "url": "https://raw.githubusercontent.com/distillpub/post--misread-tsne/f2492b469bec8aaa7aa83cc828081e14a295414d/public/assets/figure_04_original.png",
        "commit": "f2492b469bec8aaa7aa83cc828081e14a295414d",
        "license": "CC BY 2.0"
      },
      "distill-global-150-p50.png": {
        "sha256": "e9e0e405874f92faaf2fa22cec7a6e126adae0dd03a48c432aa02430577c740e",
        "url": "https://raw.githubusercontent.com/distillpub/post--misread-tsne/f2492b469bec8aaa7aa83cc828081e14a295414d/public/assets/figure_04_04.png",
        "commit": "f2492b469bec8aaa7aa83cc828081e14a295414d",
        "license": "CC BY 2.0"
      },
      "distill-global-600-input.png": {
        "sha256": "ce91aa4f8ccff0a2ebcb2f3ab62be8ddf05e18a5e46899ab873300a44aeba291",
        "url": "https://raw.githubusercontent.com/distillpub/post--misread-tsne/f2492b469bec8aaa7aa83cc828081e14a295414d/public/assets/figure_05_original.png",
        "commit": "f2492b469bec8aaa7aa83cc828081e14a295414d",
        "license": "CC BY 2.0"
      },
      "distill-global-600-p50.png": {
        "sha256": "803c7b39126c84a62fe451da36e3acd479f2e4486670a0be44293424b87d5ab5",
        "url": "https://raw.githubusercontent.com/distillpub/post--misread-tsne/f2492b469bec8aaa7aa83cc828081e14a295414d/public/assets/figure_05_04.png",
        "commit": "f2492b469bec8aaa7aa83cc828081e14a295414d",
        "license": "CC BY 2.0"
      },
      "distill-noise-input.png": {
        "sha256": "e3a26c8184a057c63c6672f5fd35f72494d287417573693e3cc4a34cafb20003",
        "url": "https://raw.githubusercontent.com/distillpub/post--misread-tsne/f2492b469bec8aaa7aa83cc828081e14a295414d/public/assets/figure_06_original.png",
        "commit": "f2492b469bec8aaa7aa83cc828081e14a295414d",
        "license": "CC BY 2.0"
      },
      "distill-noise-p2.png": {
        "sha256": "082a23937b9e57009dd54838518174fccf8129e7dd34b90624a978787f5b727f",
        "url": "https://raw.githubusercontent.com/distillpub/post--misread-tsne/f2492b469bec8aaa7aa83cc828081e14a295414d/public/assets/figure_06_01.png",
        "commit": "f2492b469bec8aaa7aa83cc828081e14a295414d",
        "license": "CC BY 2.0"
      },
      "jake-cumulative-variance.png": {
        "sha256": "d4d3e7cfd9e49d4163e13f68195609e3fe4c506751dd19d7f3e39b8949cb97d8",
        "url": "https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/d66231454ef753818dc9213c9b5942e067266966/notebooks/05.09-Principal-Component-Analysis.ipynb",
        "commit": "d66231454ef753818dc9213c9b5942e067266966",
        "license": "CC BY-NC-ND 3.0 Unported"
      },
      "jake-filtered-digits.png": {
        "sha256": "148dad1d7b80f6884fd851af3c78341520534d119b02b3ea1886eeba0a8a024f",
        "url": "https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/d66231454ef753818dc9213c9b5942e067266966/notebooks/05.09-Principal-Component-Analysis.ipynb",
        "commit": "d66231454ef753818dc9213c9b5942e067266966",
        "license": "CC BY-NC-ND 3.0 Unported"
      },
      "jake-noisy-digits.png": {
        "sha256": "2edc8f1965a03a4a1798c0470604cda70d312ea2bba28989ac62278ec702f5d7",
        "url": "https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/d66231454ef753818dc9213c9b5942e067266966/notebooks/05.09-Principal-Component-Analysis.ipynb",
        "commit": "d66231454ef753818dc9213c9b5942e067266966",
        "license": "CC BY-NC-ND 3.0 Unported"
      },
      "jake-pca-basis.png": {
        "sha256": "958095b63e56048b75dc27dfb86cf1b8c52acee3eb9ded5861130a9633065165",
        "url": "https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/d66231454ef753818dc9213c9b5942e067266966/notebooks/figures/05.09-digits-pca-components.png",
        "commit": "d66231454ef753818dc9213c9b5942e067266966",
        "license": "CC BY-NC-ND 3.0 Unported"
      },
      "jake-pixel-basis.png": {
        "sha256": "4acc90b29b21efc0abbbc268edd2a688d4a1ae9405eaaa668a1ecaa765564601",
        "url": "https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/d66231454ef753818dc9213c9b5942e067266966/notebooks/figures/05.09-digits-pixel-components.png",
        "commit": "d66231454ef753818dc9213c9b5942e067266966",
        "license": "CC BY-NC-ND 3.0 Unported"
      },
      "umap-min-dist-0.png": {
        "sha256": "dd65544c4d359197ef7775eabf5e0d0c385a43b04445b5836675eb43539ed02f",
        "url": "https://raw.githubusercontent.com/lmcinnes/umap/e82ed0d457b566b043ef44f4007a7149b0daca74/doc/images/parameters_16_1.png",
        "commit": "e82ed0d457b566b043ef44f4007a7149b0daca74",
        "license": "BSD-3-Clause"
      },
      "umap-min-dist-099.png": {
        "sha256": "446232790fc3ef130beb93c7494f9a99392c0b4bad32dc2087054cc9661ff365",
        "url": "https://raw.githubusercontent.com/lmcinnes/umap/e82ed0d457b566b043ef44f4007a7149b0daca74/doc/images/parameters_16_6.png",
        "commit": "e82ed0d457b566b043ef44f4007a7149b0daca74",
        "license": "BSD-3-Clause"
      },
      "umap-neighbors-2.png": {
        "sha256": "3bb46192a616c782c0bd6bd0ee8eb42a54565ed1f29fec940fa417bb24442ba7",
        "url": "https://raw.githubusercontent.com/lmcinnes/umap/e82ed0d457b566b043ef44f4007a7149b0daca74/doc/images/parameters_13_1.png",
        "commit": "e82ed0d457b566b043ef44f4007a7149b0daca74",
        "license": "BSD-3-Clause"
      },
      "umap-neighbors-200.png": {
        "sha256": "2cfa44a1a1057a4eac86e2a62eb12b56a32fcbe0322120a3b130d90e49b46e57",
        "url": "https://raw.githubusercontent.com/lmcinnes/umap/e82ed0d457b566b043ef44f4007a7149b0daca74/doc/images/parameters_13_7.png",
        "commit": "e82ed0d457b566b043ef44f4007a7149b0daca74",
        "license": "BSD-3-Clause"
      }
    };
    const name = entry.asset.split('/').at(-1)! as keyof typeof specifications;
    expect(specifications[name]).toBeDefined();
    const specification = specifications[name];
    expect(entry.sha256).toBe(specification.sha256);
    expect(entry.source_image_url).toBe(specification.url);
    expect(entry.commit).toBe(specification.commit);
    expect(entry.license).toBe(specification.license);
    expect(entry.source_url).toBe(specification.url
      .replace('raw.githubusercontent.com/', 'github.com/')
      .replace('/'+specification.commit+'/', '/blob/'+specification.commit+'/'));
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.crop_box_px).toBeUndefined();
    for (const [path, digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256],
      [entry.source_code,entry.source_code_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256]])
      if (path) expect(sha256(resolve(repositoryRoot,path))).toBe(digest);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual(entry.dimensions_px);
    if (entry.derivation === 'notebook-output-extraction') {
      expect(entry.source_asset_sha256).toBe('fa00993593486ca2d2445bf4781eba2ea79bb07f3360fa9456d67f573b9da9f2');
      const members: Record<string,number> = {'jake-cumulative-variance.png':34,
        'jake-noisy-digits.png':40, 'jake-filtered-digits.png':44};
      expect(entry.source_member).toBe('/cells/'+members[name]+'/outputs/0/data/image~1png');
      const notebook = JSON.parse(readFileSync(resolve(repositoryRoot,entry.source_asset!), 'utf8'));
      const data = notebook.cells[members[name]].outputs[0].data['image/png'];
      expect(readFileSync(resolve(repositoryRoot,entry.asset)))
        .toEqual(Buffer.from(Array.isArray(data)?data.join(''):data,'base64'));
    } else {
      expect(entry.derivation).toBe('source-image-copy');
      expect(entry.source_asset_sha256).toBe(entry.sha256);
    }
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/pca-tsne-umap-2026-10-25/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/pca-tsne-umap-2026-10-25/assets.json'),'utf8')) as CuratedAsset[];
    const expected=rows.find(row=>row.asset===entry.asset);
    expect(expected).toBeDefined();
    expect(entry).toEqual(expected);
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256],
      [entry.source_code,entry.source_code_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256]]) {
      if(path) expect(sha256(resolve(repositoryRoot,path))).toBe(hash);
    }
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual(entry.dimensions_px);
    if(entry.derivation==='notebook-output-extraction') {
      const notebook=JSON.parse(readFileSync(resolve(repositoryRoot,entry.source_asset!),'utf8'));
      const original=Buffer.from(notebook.cells[18].outputs[0].data['image/png'],'base64');
      expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(original);
      expect(entry.source_member).toBe('/cells/18/outputs/0/data/image~1png');
    } else expect(entry.source_asset_sha256).toBe(entry.sha256);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/spectral-2026-10-31/')) {
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/spectral-2026-10-31/assets.json'), 'utf8')) as CuratedAsset[];
    const index = rows.findIndex(row => row.asset === entry.asset);
    expect(index).toBeGreaterThanOrEqual(0);
    expect(entry).toEqual(rows[index]);
    expect(entry.sha256).toBe([
      'a5060cb9bddaaea7d3f45bfa43e25e52e9180aff0137ea2e59f75af611d2659e',
      '4615778b7c706d94775f2c2f0fc5c08ebf8e5fd12fc132879f1602500d6c3416',
    ][index]);
    expect(entry.author).toBe('Laurenz Wiskott and Fabian Schönfeld');
    expect(entry.source_asset).toBe('05 Источники/Originals/spectral-2026-10-25/wiskott-laplacian.pdf');
    expect(entry.source_asset_sha256).toBe('cb2ff89dff43e1bdcd9d3f77dea73b2b8a926a805665b28ef07fa6f6f84933fc');
    expect(entry.source_page).toBe([5, 15][index]);
    expect(entry.source_printed_page).toBe(entry.source_page);
    expect(entry.source_figure).toBe(['1.1', '2.2'][index]);
    expect(entry.render_dpi).toBe(300);
    expect(entry.render_page_size_px).toEqual([2550, 3300]);
    expect(entry.crop_box_px).toEqual([[280,980,2280,1860], [280,300,2280,1390]][index]);
    expect(entry.source_url).toBe('https://www.ini.rub.de/PEOPLE/wiskott/Teaching/UnsupervisedLearning/PublicWeb/LaplacianMatrix-L0-LectureNotes-PublicWeb.pdf#page=' + entry.source_page);
    expect(entry.license).toBe('CC BY-SA 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-sa/4.0/');
    expect(entry.license_evidence).toBe(entry.source_asset);
    expect(entry.license_evidence_sha256).toBe(entry.source_asset_sha256);
    expect(entry.derivation).toBe('source-pdf-figure-crop');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    for (const [path, digest] of [[entry.asset, entry.sha256],
      [entry.source_asset, entry.source_asset_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(digest);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot, entry.asset))))
      .toEqual([[2000, 880], [2000, 1090]][index]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/spectral-2026-10-25/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/spectral-2026-10-25/assets.json'),'utf8')) as CuratedAsset[];
    const expected=rows.find(row=>row.asset===entry.asset);
    expect(expected).toBeDefined();
    expect(entry).toEqual(expected);
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    if(entry.source_code)
      expect(sha256(resolve(repositoryRoot,entry.source_code))).toBe(entry.source_code_sha256);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset))))
      .toEqual(entry.source_page===6 ? [2000,910] : [1000,500]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/scheduling-2026-10-25/')) {
    const name = entry.asset.endsWith('/hf-dynamic-batching.png') ? 'dynamic' : 'continuous';
    const hash = name === 'dynamic'
      ? '1721080f5c46eed465bcf3d6e19f276037c7fea144acd3ec7b386a76f60ee641'
      : 'c57b1deeacea8c2e591687d1eb105722fd352492d86cdfd74c05f166fc96068d';
    const commit = 'c8c3cf72b3b724ce734259e4a086d30618a9198b';
    const source = 'https://huggingface.co/datasets/huggingface/documentation-images/';
    expect(entry.asset).toBe(`00 Учебник/Assets/Figures/curated/scheduling-2026-10-25/hf-${name}-batching.png`);
    expect(entry.id).toBe(`curated-scheduling-hf-${name}-${hash.slice(0,12)}`);
    expect(entry.author).toBe('Rémi Ouazan Reboul, Arthur Zucker, Luc Georges / Hugging Face');
    expect(entry.commit).toBe(commit);
    expect(entry.source_url).toBe(`${source}blob/${commit}/blog/continuous_batching/${name}_batching.png`);
    expect(entry.source_image_url).toBe(`${source}resolve/${commit}/blog/continuous_batching/${name}_batching.png`);
    expect(entry.source_asset).toBe(`05 Источники/Originals/scheduling-2026-10-25/${name}_batching.png`);
    expect(entry.sha256).toBe(hash);
    expect(entry.source_asset_sha256).toBe(hash);
    expect(entry.license).toBe('CC BY-NC-SA 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-nc-sa/4.0/');
    expect(entry.license_evidence).toBe('05 Источники/Originals/scheduling-2026-10-25/hf-images-README.md');
    expect(entry.license_evidence_sha256).toBe('5e6a73b7ff47813e90e2ed21d14807256b2e89fa55e5075ec69fdf01cde801ce');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    expect(entry.crop_box_px).toBeUndefined();
    for (const [path,digest] of [[entry.asset,hash], [entry.source_asset,hash],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(name==='dynamic'?[2801,1030]:[2722,2569]);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/losses-2026-10-23/')) {
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/losses-2026-10-23/assets.json'), 'utf8'));
    expect(entry).toEqual(rows.find((row:CuratedAsset) => row.asset === entry.asset));
    expect(entry.author).toBe('Andrej Karpathy');
    expect(entry.commit).toBe('8ec57ccfc32379ae4b37c45de949af359673f92f');
    expect(entry.license).toBe('MIT');
    expect(entry.license_evidence_sha256).toBe('85b365f07b4cfc5d678c548bc41965bc3bc782106b8903bd3b876d9d49a4270d');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.source_asset).toBe(entry.asset);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    for (const [path, digest] of [[entry.asset, entry.sha256],
      [entry.source_code, entry.source_code_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(digest);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/tensor-foundations-2026-10-23/')) {
    const manifest = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/tensor-foundations-2026-10-23/assets.json'), 'utf8'));
    expect(entry).toEqual(manifest.find((row: CuratedAsset) => row.asset === entry.asset));
    expect(entry.commit).toBe('0532af47d6a815298b7841de00bdbc547104b237');
    expect(entry.license).toBe('BSD-3-Clause');
    expect(entry.license_url).toBe('https://numpy.org/doc/2.3/license.html');
    expect(entry.license_evidence_sha256).toBe('1be1df33863f97a7bc1c4d67980bd6c69c9a6fef0a5ee76e6ad6cb91e56e8491');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.source_asset).toBe(entry.asset);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.crop_box_px).toBeUndefined();
    for (const [path, digest] of [[entry.asset, entry.sha256],
      [entry.license_evidence, entry.license_evidence_sha256],
      [entry.source_code, entry.source_code_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(digest);
    expect(sha256(resolve(repositoryRoot, 'site/public/licenses/numpy-v2.3.0-BSD-3-Clause.txt')))
      .toBe(entry.license_evidence_sha256);
    if (entry.asset.endsWith('.svg')) {
      expect(entry.derivation).toBe('source-svg-copy');
      const source = readFileSync(resolve(repositoryRoot, entry.asset), 'utf8');
      expect(source).toContain('viewBox=');
      expect(source).not.toContain('<image');
    } else {
      expect(entry.derivation).toBe('source-raster-copy');
      expectDecodableImage(resolve(repositoryRoot, entry.asset));
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual(
        (entry as unknown as {dimensions_px:number[]}).dimensions_px);
      expect(readFileSync(resolve(repositoryRoot, entry.source_code!), 'utf8'))
        .toContain('Image credits: Jay Alammar');
    }
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/preference-2026-10-23/hh-')
    || entry.asset.startsWith('00 Учебник/Assets/Figures/curated/reward-modeling-2026-10-23/')) {
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,
      entry.asset.slice(0, entry.asset.lastIndexOf('/')) + '/assets.json'), 'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows.find(row => row.asset === entry.asset));
    expect(entry.author).toBe('Yuntao Bai et al.');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.license_evidence_sha256).toBe('9f30f8807321128513e1aeef97a7d52e0daf874802c15d8b7d54feadfef10d02');
    const originals: Record<string, [string, number, number[], number[]]> = {
      'hh-workflow.png': ['f1b6b64ba26f7ec28cd54d80013c7e769153324705b409c7ff6acb74930d16fe',
        5, [350, 300, 2200, 1230], [1850, 930]],
      'hh-feedback-interface.png': ['1e05a370a1ce3f1e1b9a742fcb406a790b6d0a61856ef02147e4d15605febca1',
        10, [350, 270, 2200, 1770], [1850, 1500]],
      'hh-data-distributions.png': ['54e1c75340ea96895d8b0810fbf6fc82edd0036c0ba8a003d607c4ec389f0eb5',
        21, [350, 270, 2200, 1350], [1850, 1080]],
      'hh-calibration.png': ['74e8528aedab2bb961f831beddbb34c7ed8ad34cfecbc7ab8a2d5fb4c2a31d80',
        14, [350, 300, 2200, 1150], [1850, 850]],
    };
    const original = originals[entry.asset.split('/').at(-1)!];
    expect(original).toBeTruthy();
    expect(entry.sha256).toBe(original[0]);
    expect(entry.source_url).toBe('https://arxiv.org/pdf/2204.05862v1#page=' + original[1]);
    expect(entry.source_asset_sha256).toBe('26e390d4f76938d2e2591225603637c643cc108e94afb756c65f5d2e0fc9037b');
    expect(entry.source_page).toBe(original[1]);
    expect(entry.render_dpi).toBe(300);
    expect(entry.render_page_size_px).toEqual([2550, 3300]);
    expect(entry.crop_box_px).toEqual(original[2]);
    expect(entry.derivation).toBe('pdf-page-render-crop');
    for (const [file, digest] of [[entry.asset, entry.sha256],
      [entry.source_asset, entry.source_asset_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(digest);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual(original[3]);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/sft-2026-10-23/minicpm-two-stage-mixture.png') {
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/sft-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(entry).toEqual(rows[1]);
    expect(entry.sha256).toBe('700ff000c1999eb13c98de77a33fb4fc77c6d9c0de072997a590c1aa2a5bd2c3');
    expect(entry.source_asset_sha256).toBe('46bd6d935ba60d5dec6ab4a1bfe9fbce6a8b81a03be9dd91c4502f4aabe5b880');
    expect(entry.source_url).toBe('https://arxiv.org/pdf/2404.06395v1#page=12');
    expect(entry.license_evidence_sha256).toBe('e9c7e2b7585330c5f6e57e593b81e391f838e6acf3ae6c218bfb750a1cdb4266');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.source_page).toBe(12);
    expect(entry.render_dpi).toBe(600);
    expect(entry.render_page_size_px).toEqual([5100, 6600]);
    expect(entry.crop_box_px).toEqual([860, 3380, 4240, 4980]);
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([3380, 1600]);
    for (const [file, digest] of [[entry.asset, entry.sha256],
      [entry.source_asset, entry.source_asset_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain('creativecommons.org/licenses/by/4.0/');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/sft-2026-10-23/')) {
    const [spec] = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/sft-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(entry).toEqual(spec);
    expect(entry.sha256).toBe('c41ba1e6892e3cdc3bb1a3abca66e73d096e4af2f70779cf1c31b0cb0b3dd545');
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.source_url).toBe('https://arxiv.org/html/2411.15124v5#S0.F1');
    expect(entry.source_image_url).toBe('https://arxiv.org/html/2411.15124v5/tulu3_figure1_v2.png');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.license_evidence_sha256).toBe('0a437403a0fce0cab42931983f67753c1260df36df66dbd34abfe4603e9ef0ef');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.crop_box_px).toBeUndefined();
    for (const [file, digest] of [[entry.asset, entry.sha256],
      [entry.source_asset, entry.source_asset_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot, entry.asset))).toEqual(
      readFileSync(resolve(repositoryRoot, entry.source_asset!)));
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain('License: CC BY 4.0');
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([2040, 712]);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.provenance_confirmation === 'pinned-hf-dataset-animation') {
    const pin = 'a9c3851af92cce883828811163f904b61a2550dc';
    const match = entry.source_url?.match(new RegExp('^https://huggingface\\.co/datasets/huggingface/documentation-images/blob/'
      + pin + '/ppl_(full|chunked|sliding)\\.gif$'));
    expect(match).toBeTruthy();
    const name = `ppl-${match?.[1]}.gif`;
    expect(entry.asset).toBe('00 Учебник/Assets/Figures/curated/ntp-2026-10-23/' + name);
    expect(entry.source_asset).toBe('05 Источники/Originals/ntp-2026-10-23/' + name);
    expect(entry.source_image_url).toBe(entry.source_url?.replace('/blob/', '/resolve/'));
    expect(entry.commit).toBe(pin);
    expect(entry.license).toBe('CC BY-NC-SA 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-nc-sa/4.0/');
    expect(entry.license_evidence).toBe('05 Источники/Originals/ntp-2026-10-23/hf-images-README.md');
    expect(entry.license_evidence_sha256).toBe('5e6a73b7ff47813e90e2ed21d14807256b2e89fa55e5075ec69fdf01cde801ce');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.author).toBe('Hugging Face Team');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.crop_box_px).toBeUndefined();
    requireNonEmptyString(entry.modifications, 'unchanged animation description');
    for (const [path, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]]) {
      expect(hash).toMatch(/^[a-f0-9]{64}$/);
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    }
    const bytes = readFileSync(resolve(repositoryRoot, entry.asset));
    expect(bytes).toEqual(readFileSync(resolve(repositoryRoot, entry.source_asset!)));
    expect([bytes.readUInt16LE(6), bytes.readUInt16LE(8)]).toEqual([1200, 160]);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/sequence-alternatives-2026-10-23/')){
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,'publishing/tools/sequence-figures.json'),'utf8'));
    const spec=specs.find((s:{asset:string})=>s.asset===entry.asset);
    expect(spec).toBeDefined();
    expect(spec.source_sha256).toBe('6fe16414fa471a5ab150cddcba5840d3a94e347e96992c50c12c1fb8b919479f');
    for(const [key,value] of Object.entries({
      author:'Tri Dao and Albert Gu',source_url:spec.source_url,
      source_asset:spec.source,source_asset_sha256:spec.source_sha256,
      source_page:spec.page,render_dpi:spec.dpi,crop_box_px:spec.crop_pixels,
      render_page_size_px:[2550,3300],sha256:spec.sha256,
      license:'CC BY 4.0',license_url:'https://creativecommons.org/licenses/by/4.0/',
      license_evidence:'05 Источники/Originals/sequence-alternatives-2026-10-23/mamba2-v1.html',
      license_evidence_sha256:'42977be717723e2de308840e89846fca68f8f2c072e3330d5a2305a24d18589d',
      provenance_confirmation:'source-license-verified',derivation:'pdf-page-render-crop'
    }))expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    for(const [path,digest] of [[entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8')).toContain('License: CC BY 4.0');
    expect(entry.modifications).toContain('full English caption retained');
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(spec.dimensions);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  const retainedMoe:Record<string,{name:string;hash:string;size:number[]}>={
    '00 Учебник/Assets/Figures/curated/source-first-37-40-audit/megablocks-expert-matmuls.png':
      {name:'megablocks-expert-matmuls.png',hash:'f9176ec2fbce60e070cb5d431ed3cc2aa5080cde43cb957048992dd875502e7c',size:[1136,324]},
    '00 Учебник/Assets/Figures/curated/moe-review-2026-09/standard-moe.png':
      {name:'standard-moe.png',hash:'4faa85d7ce72bf4958d535f315d39f7cce0cf487658ff96ce104b806033bd8b6',size:[414,535]},
    '00 Учебник/Assets/Figures/curated/moe-review-2026-09/latent-moe.png':
      {name:'latent-moe.png',hash:'1bfcacc1cfe641d4f5f46d174fc9b31c12d84022dfaadd003c5b3106872bc13b',size:[518,615]},
  };
  const retained=retainedMoe[entry.asset];
  if(retained){
    expect(entry.source_asset).toBe('05 Источники/Originals/moe-2026-10-23/'+retained.name);
    expect(entry.source_asset_sha256).toBe(retained.hash);
    expect(entry.sha256).toBe(retained.hash);
    expect(entry.derivation).toBe('source-raster-copy');
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(retained.hash);
    expect(sha256(resolve(repositoryRoot,entry.source_asset!))).toBe(retained.hash);
    expect(readFileSync(resolve(repositoryRoot,entry.asset)).equals(readFileSync(resolve(repositoryRoot,entry.source_asset!)))).toBe(true);
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(retained.size);
    if(retained.name.startsWith('megablocks')){
      expect(entry.author).toBe('Trevor Gale et al.');
      expect(entry.source_url).toBe('https://huggingface.co/blog/moe');
      expect(entry.source_image_url).toBe('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/11_expert_matmuls.png');
      expect(entry.license).toBe('open-educational-web-material; reuse confirmed by repository owner');
      expect(entry.provenance_confirmation).toBe('user-confirmed-open-materials');
    }else{
      expect(entry.author).toBe('Venmugil Elango et al.');
      expect(entry.source_url).toBe('https://arxiv.org/abs/2601.18089v1');
      expect(entry.source_image_url).toBe('https://arxiv.org/html/2601.18089v1/figures/'+retained.name.replaceAll('-','_'));
      expect(entry.license).toBe('CC BY 4.0');
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
      expect(entry.provenance_confirmation).toBe('source-license-verified');
    }
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/moe-2026-10-23/')){
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,'publishing/tools/moe-figures.json'),'utf8'));
    const spec=specs.find((s:{asset:string})=>s.asset===entry.asset);
    expect(spec).toBeDefined();
    for(const [key,value] of Object.entries(spec))
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    for(const [path,digest] of [[entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    if(spec.crop_box_px){
      expect(entry.license).toBe('CC BY 4.0');
      expect(entry.license_evidence).toBe(entry.source_asset);
      expect(entry.modifications).toContain('full English caption retained');
      const [x1,y1,x2,y2]=spec.crop_box_px;
      expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual([x2-x1,y2-y1]);
    }else{
      expect(entry.license).toBe('Apache-2.0');
      expect(entry.commit).toBe('357454f4f647385839c0ff6b99a688dc7cd9c13f');
      expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8')).toContain('Apache License');
      expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(spec.render_page_size_px);
    }
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if(entry.asset==='00 Учебник/Assets/Figures/curated/modern-37-40-moe/mixtral-smoe-layer.png'){
    expect(entry.author).toBe('Albert Q. Jiang et al. / Mistral AI');
    expect(entry.source_url).toBe('https://arxiv.org/html/2401.04088v1#S2.F1');
    expect(entry.source_image_url).toBe('https://arxiv.org/html/2401.04088v1/images/smoe.png');
    expect(entry.source_asset).toBe('05 Источники/Originals/ffn-2026-10-23/mixtral-smoe-original.png');
    expect(entry.source_asset_sha256).toBe('1347eb75040f1144f7ea5820cae7b4bd60c49b6bf351fb1c9e77b4c20e642c46');
    expect(entry.sha256).toBe(entry.source_asset_sha256);
    expect(entry.source_page_snapshot).toBe('05 Источники/Originals/ffn-2026-10-23/mixtral-v1.html');
    expect(entry.source_page_sha256).toBe('9f3a55b554c6216cb9fbdce3f9b3e7b3ddad44b97ce1326e8c23f753e74fba02');
    expect(entry.license_evidence).toBe('05 Источники/Originals/ffn-2026-10-23/mixtral-v1-abs.html');
    expect(entry.license_evidence_sha256).toBe('f64f17740fbeca1bad2406c8b2402c24c019011deb0c1c451eef2beecad6a226');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('source-raster-copy');
    for(const [path,digest] of [[entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],[entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8')).toContain('creativecommons.org/licenses/by/4.0/');
    expect(readFileSync(resolve(repositoryRoot,entry.asset)).equals(readFileSync(resolve(repositoryRoot,entry.source_asset)))).toBe(true);
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual([2116,968]);
    return;
  }
  if(entry.asset==='00 Учебник/Assets/Figures/curated/ffn-2026-10-23/geva-composition.svg'){
    expect(entry.source_url).toBe('https://arxiv.org/html/2012.14913v1#S5.F8');
    expect(entry.source_image_url).toBe('https://arxiv.org/html/2012.14913v1/ffn_prediction_zero_agreement.svg');
    expect(entry.source_asset).toBe('05 Источники/Originals/ffn-2026-10-23/geva-composition.svg');
    expect(entry.source_page_snapshot).toBe('05 Источники/Originals/ffn-2026-10-23/geva-v1.html');
    expect(entry.source_page_sha256).toBe('30f5193d6550d1af3ff7a321baebe1abda8da6acf6849987941df99fd55d0f38');
    expect(entry.license_evidence).toBe('05 Источники/Originals/ffn-2026-10-23/geva-v1-abs.html');
    expect(entry.license_evidence_sha256).toBe('bcbff810ca7753b9e6b72362b4553e5587f81dbc21f0916c38de5019dec781d0');
    expect(entry.sha256).toBe('dd5f47d1c4e9b18830c7153ff10acea4dd8e83977d3ec471d33269582d464eee');
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.author).toBe('Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy');
    expect(entry.license).toBe('CC0 1.0');
    expect(entry.license_url).toBe('https://creativecommons.org/publicdomain/zero/1.0/');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('source-svg-copy');
    for(const [path,digest] of [[entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],[entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8')).toContain('creativecommons.org/publicdomain/zero/1.0/');
    expect(readFileSync(resolve(repositoryRoot,entry.asset)).equals(readFileSync(resolve(repositoryRoot,entry.source_asset)))).toBe(true);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/long-context-2026-10-23/')){
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,'publishing/tools/long-context-figures.json'),'utf8'));
    const spec=specs.find((s:{asset:string})=>s.asset===entry.asset);
    expect(spec).toBeDefined();
    for(const [key,value] of Object.entries(spec))
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    for(const [path,digest] of [[entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8'))
      .toContain('creativecommons.org/licenses/by/4.0/');
    expect(entry.source_image_url).toBe(entry.source_url);
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.derivation).toBe('pdf-crop');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.modifications).toContain('English caption');
    const [x1,y1,x2,y2]=spec.crop_box_px;
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual([x2-x1,y2-y1]);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  const mlaSpecs=JSON.parse(readFileSync(resolve(repositoryRoot,'publishing/tools/mla-figures.json'),'utf8'));
  const mlaSpec=mlaSpecs.find((s:{asset:string})=>s.asset===entry.asset);
  if(mlaSpec){
    for(const [key,value] of Object.entries(mlaSpec))
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    for(const [path,digest] of [[entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    if(mlaSpec.crop_box_px){
      for(const [path,digest] of [[entry.license_evidence,entry.license_evidence_sha256],
        [entry.source_page_snapshot,entry.source_page_sha256]])
        expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
      expect(entry.derivation).toBe('source-raster-crop');
      const [x1,y1,x2,y2]=mlaSpec.crop_box_px;
      expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual([x2-x1,y2-y1]);
      const readme=readFileSync(resolve(repositoryRoot,entry.source_page_snapshot!),'utf8');
      expect(readme).toContain('This code repository is licensed under [the MIT License](LICENSE-CODE)');
      expect(readme).toContain('/figures/architecture.png?raw=true');
      expect(entry.modifications).toContain('not asserted for separate paper figures');
    }else{
      expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(
        readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    }
    if(mlaSpec.rights_status==='paper-figure-license-unverified'){
      expect(entry.license).toBe('Figure reuse terms unconfirmed; arXiv non-exclusive distribution is not a reuse license');
      expect(entry.provenance_confirmation).toBe('user-confirmed-open-materials');
    }else expect(entry.provenance_confirmation).toBe('source-license-verified');
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/gqa-2026-10-22/')){
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,'publishing/tools/gqa-figures.json'),'utf8'));
    const spec=specs.find((s:{asset:string})=>s.asset===entry.asset);
    expect(spec).toBeDefined();
    for(const [key,value] of Object.entries(spec))
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    expect(entry.author).toBe('Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, Sumit Sanghai');
    expect(entry.source_url).toBe('https://aclanthology.org/2023.emnlp-main.298.pdf#page='+spec.source_page);
    expect(entry.source_image_url).toBe(entry.source_url);
    expect(entry.source_printed_page).toBe(spec.source_page+4894);
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.license_evidence).toBe('05 Источники/Originals/gqa-2026-10-22/acl-copyright.html');
    expect(entry.license_evidence_sha256).toBe('88d644d2e21542ef1cc6c304c264ce7543d81e87c9599aff7e1c68e7a6df9558');
    for(const [path,digest] of [[entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(entry.derivation).toBe('pdf-crop');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.modifications).toContain('English caption');
    const [x1,y1,x2,y2]=spec.crop_box_px;
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual([x2-x1,y2-y1]);
    return;
  }
  if(entry.asset.startsWith('00 Учебник/Assets/Figures/curated/transformer-shapes-2026-10-22/')){
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,'publishing/tools/transformer-shapes-figures.json'),'utf8'));
    const spec=specs.find((s:{asset:string})=>s.asset===entry.asset);
    expect(spec).toBeDefined();
    for(const [key,value] of Object.entries(spec))
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    const page=spec.source_page;
    const commit=page?'a158843b20107949f1a8d7df1b05cd33b9166712':'173fb0851d7e654872f0327aff2e17607cf24106';
    const repo=page?'stanford-cs336/assignment1-basics':'jax-ml/scaling-book';
    expect(entry.author).toBe(page?'Stanford CS336 staff':'Jacob Austin et al., How To Scale Your Model');
    expect(entry.commit).toBe(commit);
    expect(entry.source_url).toBe('https://github.com/'+repo+'/blob/'+commit
      +(page?'/cs336_assignment1_basics.pdf#page='+page:'/assets/img/roofline-plot-q3.png'));
    expect(entry.license).toBe('MIT');
    expect(entry.license_url).toBe('https://github.com/'+repo+'/blob/'+commit+'/LICENSE');
    expect(entry.license_evidence).toBe(page
      ?'05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment1-basics/LICENSE'
      :'05 Источники/Originals/transformer-shapes-2026-10-22/jax-LICENSE');
    expect(entry.license_evidence_sha256).toBe(page
      ?'b3ccacaf9286910f22ba98681601f426288297f17004699f99d9fdd5f7432751'
      :'2567a694c350d7bd2e509ae31062341ff74d59f4f5b1b3dc49ca32dd79570ebb');
    for(const [path,digest] of [[entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.derivation).toBe(page?'pdf-page-render':'source-raster-copy');
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(page?[1530,1980]:[2478,1290]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/rope-2026-10-22/')) {
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,'publishing/tools/rope-figures.json'),'utf8'));
    const spec=specs.find((s:{asset:string})=>s.asset===entry.asset);
    expect(spec).toBeDefined();
    for(const [key,value] of Object.entries(spec))
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    const pi=entry.asset.endsWith('/pi-extrapolation.png');
    expect(entry.author).toBe(pi
      ? 'Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian'
      : 'Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu');
    expect(entry.source_url).toBe(pi
      ? 'https://arxiv.org/pdf/2306.15595v2#page=4'
      : 'https://arxiv.org/pdf/2104.09864v5#page=8');
    expect(entry.license).toBe(pi?'CC BY 4.0':'CC BY-NC-ND 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/'+(pi?'by':'by-nc-nd')+'/4.0/');
    expect(entry.license_evidence).toBe(pi
      ? '05 Источники/Originals/rope-2026-10-22/pi-abstract.html'
      : '05 Источники/Originals/positional-information-2026-10-21/roformer-v5-abstract.html');
    expect(entry.license_evidence_sha256).toBe(pi
      ? '47044b77296045d4fe89747384b31b568da898e043e5355783e58f1233f6ef9b'
      : 'aa47247bbd7f9b18970e8f6adc201cb641ff7bd461ba7e2814042f15259d9043');
    for(const [path,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],[entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    const [x1,y1,x2,y2]=entry.crop_box_px as number[];
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual([x2-x1,y2-y1]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/norm-gating-2026-10-22/')) {
    const specs=JSON.parse(readFileSync(resolve(repositoryRoot,'publishing/tools/norm-gating-figures.json'),'utf8'));
    const spec=specs.find((s:{asset:string})=>s.asset===entry.asset);
    expect(spec).toBeDefined();
    for(const [key,value] of Object.entries(spec))if(key!=='name')
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    expect(entry.source_url).toBe('https://proceedings.mlr.press/v119/xiong20b.html');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.license_evidence).toBe('05 Источники/Originals/norm-gating-2026-10-22/pmlr-license-agreement.html');
    expect(entry.license_evidence_sha256).toBe('1c8123a6c840ca2752e383f0924fa23ba38552140a2cc54f69ea9df5acfcdd3d');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    for(const [path,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],[entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!),'utf8'))
      .toContain('Creative Commons Attribution 4.0 International');
    const [x1,y1,x2,y2]=entry.crop_box_px as number[];
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual([x2-x1,y2-y1]);
    return;
  }
  if (entry.asset==='00 Учебник/Assets/Figures/curated/source-first-32-34/rmsnorm-paper-results.png') {
    expect(entry.metadata_status).toBe('needs-confirmation');
    expect(entry.rights_status).toBe('license-unverified; incomplete-caption-crop-withdrawn');
    expect((entry as unknown as {used_in:string[]}).used_in).toEqual([]);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  // The source PDF and full-slide pixels are verified. Copies are retained
  // locally for that check, but are no longer used by the published chapter.
  const pendingSlidePrefix = '00 Учебник/Assets/Figures/curated/foundations-source-first/';
  const pendingSlides: Record<string, {id: string; hash: string; page: number; printed: number}> = {
    'cs231n-softmax-cross-entropy.png': {
      id: 'curated-cs231n-softmax-ce-0000000000a6',
      hash: '48e8acfe3a1c34d5b8bc5efbe89fa57cb0a8d9aa07cab18acc902be9d2ffcecb',
      page: 75, printed: 77
    },
    'cs231n-linear-decision-boundaries.png': {
      id: 'curated-cs231n-boundaries-0000000000a7',
      hash: '1c5ac40bbb84b0d962ef5ed15a6e3c63097fd8f6ccd3535a20f2cf9d4a33b7e6',
      page: 61, printed: 63
    }
  };
  const pendingSlide = entry.asset.startsWith(pendingSlidePrefix)
    ? pendingSlides[entry.asset.slice(pendingSlidePrefix.length)] : undefined;
  if (pendingSlide) {
    expect(entry.id).toBe(pendingSlide.id);
    expect(entry.source_url).toBe('https://cs231n.stanford.edu/slides/2025/lecture_2.pdf');
    expect(entry.author).toBe('Stanford CS231n; Ehsan Adeli (lecturer, 2025-04-03)');
    expect((entry as unknown as {author_evidence_url:string}).author_evidence_url)
      .toBe('https://cs231n.stanford.edu/2025/schedule.html');
    expect(entry.source_asset).toBe('05 Источники/Originals/cs231n-provenance-2026-10-29/lecture_2.pdf');
    expect(entry.source_asset_sha256).toBe('191488d0a96f85574e8e2ebb8246865d3aac6d84da88fe0962aa1651f814565e');
    expect(sha256(resolve(repositoryRoot,entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('pdf-full-slide-render-verified');
    expect(entry).toMatchObject({
      publication_status: 'retired', license: 'Unverified for redistribution', license_url: null,
      rights_status: 'course-slide-republication-unverified', used_in: [],
      former_used_in: ['00 Учебник/00 Математические и ML-основания/04 Функции потерь.md'],
    });
    expect(entry.provenance_confirmation).toBe('source-pixels-verified');
    expect(entry.render_dpi).toBe(160);
    const receipt=JSON.parse(readFileSync(resolve(repositoryRoot,'docs/audit/2026-10-29-cs231n-source-pixels.json'),'utf8'));
    const pixelRow=receipt.records.find((r:{id:string})=>r.id===entry.id);
    expect(receipt.source_asset_sha256).toBe(entry.source_asset_sha256);
    expect(pixelRow.sha256).toBe(entry.sha256);
    expect(pixelRow.identical_rgb_pixels).toBe(true);
    expect(pixelRow.mean_absolute_error_rgb).toEqual([0,0,0]);
    expect(entry.source_page).toBe(pendingSlide.page);
    expect(entry.source_printed_page).toBe(pendingSlide.printed);
    expect(entry.sha256).toBe(pendingSlide.hash);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(pendingSlide.hash);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([1600, 900]);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  const retiredSpec=retiredFigureCopies.find((s:{asset:string})=>s.asset===entry.asset);
  if(retiredSpec){
    const row=entry as unknown as Record<string,unknown>;
    expect(row.publication_status).toBe('retired');
    expect(row.used_in).toEqual(retiredSpec.current_usage);
    expect(row.former_used_in).toEqual(retiredSpec.former_used_in);
    expect(entry.metadata_status).toBe('needs-confirmation');
    expect(entry.rights_status).toBe('paper-figure-license-unverified');
    for(const [key,value] of Object.entries(retiredSpec)){
      if(['name','width','height','frames','used_in','current_usage','former_used_in'].includes(key))continue;
      expect(row[key]).toEqual(value);
    }
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(retiredSpec.sha256);
    expect(sha256(resolve(repositoryRoot,entry.source_asset!))).toBe(retiredSpec.source_asset_sha256);
    expect(pngDimensions(resolve(repositoryRoot,entry.asset)))
      .toEqual([retiredSpec.width,retiredSpec.height]);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  const gpt3Spec=gpt3Figures.find((s:{asset:string})=>s.asset===entry.asset);
  if(gpt3Spec){
    for(const [key,value] of Object.entries(gpt3Spec)){
      if(['name','width','height','frames'].includes(key))continue;
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    }
    for(const [path,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    if(entry.rights_status==='source-license-verified'){
      for(const [path,digest] of [[entry.source_page_snapshot,entry.source_page_sha256],
        [entry.license_evidence,entry.license_evidence_sha256]])
        expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
      if(entry.derivation==='source-gif-copy')
        expect(readFileSync(resolve(repositoryRoot,entry.asset)).equals(
          readFileSync(resolve(repositoryRoot,entry.source_asset!)))).toBe(true);
      else expect(entry.derivation).toBe('source-gif-frame-export');
    }else{
      expect(entry.rights_status).toBe('paper-figure-license-unverified');
      expect(entry.metadata_status).toBe('needs-confirmation');
    }
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/agent-approval-2026-11-09/microsoft-transactional-outbox.svg') {
    expect(entry.id).toBe('curated-microsoft-transactional-outbox-36bdda0a5038');
    expect(entry.author).toBe('Microsoft Azure Architecture Center');
    expect(entry.source_url).toBe('https://github.com/MicrosoftDocs/architecture-center/blob/629acf25fc1ab4096cfa102367a569867899b79e/docs/databases/guide/_images/out-box-pattern.svg');
    expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/MicrosoftDocs/architecture-center/629acf25fc1ab4096cfa102367a569867899b79e/docs/databases/guide/_images/out-box-pattern.svg');
    expect(entry.commit).toBe('629acf25fc1ab4096cfa102367a569867899b79e');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.source_asset).toBe(entry.asset);
    expect(entry.sha256).toBe('36bdda0a5038eb48682f6b0057141e21b2963a0379b1025d46834b2e7af7262a');
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.source_page_snapshot).toBe('05 Источники/Originals/agent-approval-2026-11-09/outbox.md');
    expect(entry.source_page_sha256).toBe('a2864017a09d8559659f5b4407aafc303c1248477fc55f6ebcf8d6843d969bf9');
    expect(entry.license_evidence).toBe('05 Источники/Originals/agent-approval-2026-11-09/MicrosoftDocs-CC-BY-4.0.txt');
    expect(entry.license_evidence_sha256).toBe('7183ef9749eb9b75e710a81a4c812a9405cc184f087500581a973e5929ac3f5e');
    for (const [file, hash] of [[entry.asset, entry.sha256],
      [entry.source_page_snapshot, entry.source_page_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(entry.source_viewbox).toEqual([0, 0, 628.563, 515.106]);
    expect(entry.crop_box_px).toBeUndefined();
    const svg = readFileSync(resolve(repositoryRoot, entry.asset), 'utf8');
    expect(svg).toContain('viewBox="0 0 628.563 515.106"');
    expect(svg).toContain('Commit transaction');
    expect(svg).toContain('Publish events');
    return;
  }
  const natureFigure = {
    '00 Учебник/Assets/Figures/curated/alphaproof-nature/figure-4-search-vs-ttrl.png': {
      id: 'curated-alphaproof-nature-e0474a1f3a96',
      url: 'https://www.nature.com/articles/s41586-025-09833-y/figures/4',
      image: 'https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-025-09833-y/MediaObjects/41586_2025_9833_Fig4_HTML.png',
      author: 'Thomas Hubert, Rishi Mehta, Laurent Sartran et al.',
      sha: 'e0474a1f3a96dd82346cb66f8761da69b9357299c72b6d04e0c673069f86e1cc',
      evidence: '05 Источники/Originals/alphaproof-nature/pmc12999475.xml',
      evidenceSha: '70799ee4b8c683b592583357d1d349d4e6827474c08c04dead02d6af482c5870',
      size: [1766, 646],
    },
    '00 Учебник/Assets/Figures/curated/program-search-2026-11-09/funsearch-figure-1.png': {
      id: 'curated-funsearch-program-search-72cf2e2bcf47',
      url: 'https://www.nature.com/articles/s41586-023-06924-6/figures/1',
      image: 'https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-023-06924-6/MediaObjects/41586_2023_6924_Fig1_HTML.png',
      author: 'Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov et al.',
      sha: '72cf2e2bcf4723c455d001ed30e062402356434e93d98c0637aaa513e34f7bb1',
      evidence: '05 Источники/Originals/program-search-2026-11-09/funsearch.xml',
      evidenceSha: '6c512b475e85a5f9be1208478934b38facef6d0f20e285a72b02eedfad2e4f81',
      size: [1327, 687],
    },
  }[entry.asset];
  if (natureFigure) {
    expect(entry.id).toBe(natureFigure.id);
    expect(entry.source_url).toBe(natureFigure.url);
    expect(entry.source_image_url).toBe(natureFigure.image);
    expect(entry.author).toBe(natureFigure.author);
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.source_asset).toBe(entry.asset);
    expect(entry.sha256).toBe(natureFigure.sha);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual(natureFigure.size);
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.license_evidence).toBe(natureFigure.evidence);
    expect(entry.license_evidence_sha256).toBe(natureFigure.evidenceSha);
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).toContain('https://creativecommons.org/licenses/by/4.0/');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  const gpt2Spec=gpt2Figures.find((s:{asset:string})=>s.asset===entry.asset);
  if(gpt2Spec){
    for(const [key,value] of Object.entries(gpt2Spec)){
      if(['name','width','height','frames'].includes(key))continue;
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    }
    for(const [path,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(
      readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/source-audit-25-31/gpt1-figure1.png') {
    expect(entry.id).toBe('curated-gpt1-figure1-0000000000c9');
    expect(entry.author).toBe('Alec Radford et al.');
    expect(entry.source_url).toBe('https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf');
    expect(entry.source_asset).toBe('05 Источники/Originals/full-transformer-2026-10-21/gpt1.pdf');
    expect(entry.source_asset_sha256).toBe('eb81a65e0856bd38e855e1f1de6ee12e7c3eb92ae05c8270d345a475beb631d0');
    expect(entry.sha256).toBe('0d3480a98ca4f38b69a93bb750c3a70396f333e200bafe2fec7b5deb42327e8c');
    expect(entry.license).toBe('unverified; inherited paper figure, source verified separately');
    expect(entry.metadata_status).toBe('needs-confirmation');
    expect(entry.rights_status).toBe('paper-figure-license-unverified');
    expect(entry.provenance_confirmation).toBe('user-confirmed-open-materials');
    expect(entry.source_page).toBe(4);
    expect(entry.render_dpi).toBe(360);
    expect(entry.render_page_size_px).toEqual([3060,3960]);
    expect(entry.crop_box_px).toEqual([500,300,2550,1150]);
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(sha256(resolve(repositoryRoot,entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(entry.sha256);
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual([2050,850]);
    validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset)));
    return;
  }
  const architectureSpec = [...architectureFigures,...t5Figures,...llamaFigures,...bertFamilyFigures.filter(
    (s: {provenance_confirmation: string})=>s.provenance_confirmation==='bert-family-source-snapshot')]
    .find((s: {asset: string}) => s.asset === entry.asset);
  if (architectureSpec) {
    for (const [key, value] of Object.entries(architectureSpec)) {
      if (['name','width','height'].includes(key)) continue;
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    }
    for (const [path,hash] of [
      [entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]
    ]) expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(
      [architectureSpec.width,architectureSpec.height]);
    if (entry.derivation === 'source-png-copy')
      expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(
        readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset)));
    return;
  }
  const fullSpec = fullTransformerFigures.find((s: {asset: string}) => s.asset === entry.asset);
  if (fullSpec) {
    for (const key of ['author','source_url','source_image_url','license','license_url',
      'source_asset','source_page_snapshot','source_page_sha256','sha256'])
      expect((entry as unknown as Record<string,unknown>)[key]).toBe(fullSpec[key]);
    expect(entry.source_asset_sha256).toBe(fullSpec.sha256);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('full-transformer-source-snapshot');
    expect(entry.license_evidence).toBe(fullSpec.source_page_snapshot);
    expect(entry.license_evidence_sha256).toBe(fullSpec.source_page_sha256);
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(fullSpec.sha256);
    expect(sha256(resolve(repositoryRoot,entry.source_asset!))).toBe(fullSpec.sha256);
    expect(sha256(resolve(repositoryRoot,entry.license_evidence!))).toBe(fullSpec.source_page_sha256);
    expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(
      readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (Object.hasOwn(positionalFigures, entry.asset)) {
    const s = positionalFigures[entry.asset as keyof typeof positionalFigures];
    for (const [key,value] of Object.entries(s.fields))
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    for (const [path,hash] of [
      [entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]
    ]) expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    if (entry.derivation?.startsWith('source-'))
      expect(readFileSync(resolve(repositoryRoot,entry.asset)).equals(readFileSync(resolve(repositoryRoot,entry.source_asset!)))).toBe(true);
    if ('viewBox' in s)
      expect(readFileSync(resolve(repositoryRoot,entry.asset),'utf8')).toContain('viewBox="'+s.viewBox+'"');
    else if (entry.asset.endsWith('.png')) {
      expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(s.size);
      validateRenderedPngBytes(readFileSync(resolve(repositoryRoot,entry.asset)));
    }
    if ('public_license_copy' in s.fields)
      expect(readFileSync(resolve(repositoryRoot,s.fields.public_license_copy)).equals(readFileSync(resolve(repositoryRoot,entry.license_evidence!)))).toBe(true);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (Object.hasOwn(maskingHeadsFigures, entry.asset)) {
    const s = maskingHeadsFigures[entry.asset as keyof typeof maskingHeadsFigures];
    for (const [key,value] of Object.entries(s.fields))
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    for (const [path,hash] of [
      [entry.asset,entry.sha256], [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256]
    ]) expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    if ('license_evidence' in s.fields)
      expect(sha256(resolve(repositoryRoot,entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(readFileSync(resolve(repositoryRoot,entry.asset)).equals(readFileSync(resolve(repositoryRoot,entry.source_asset!)))).toBe(true);
    expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(s.size);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (Object.hasOwn(selfAttentionFigures, entry.asset)) {
    const s = selfAttentionFigures[entry.asset as keyof typeof selfAttentionFigures];
    for (const [key,value] of Object.entries(s.fields))
      expect((entry as unknown as Record<string,unknown>)[key]).toEqual(value);
    for (const [path,hash] of [
      [entry.asset,entry.sha256], [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]
    ]) expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot,entry.asset)).equals(readFileSync(resolve(repositoryRoot,entry.source_asset!)))).toBe(true);
    if ('size' in s) expect(pngDimensions(resolve(repositoryRoot,entry.asset))).toEqual(s.size);
    else expect(readFileSync(resolve(repositoryRoot,entry.asset),'utf8')).toContain('viewBox="'+s.viewBox+'"');
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (Object.hasOwn(attentionTranslationFigures, entry.asset)) {
    const s = attentionTranslationFigures[entry.asset as keyof typeof attentionTranslationFigures];
    expect(entry.id).toBe(s.id);
    expect(entry.author).toBe(s.author);
    expect(entry.source_url).toBe(s.url);
    expect(entry.source_image_url).toBe(s.image);
    expect(entry.source_asset).toBe(s.source);
    expect(entry.source_asset_sha256).toBe(s.sourceHash);
    expect(entry.sha256).toBe(s.hash);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(s.sourceHash);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(s.hash);
    expect(entry.source_page_snapshot).toBe(s.page);
    expect(entry.source_page_sha256).toBe(s.pageHash);
    expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(s.pageHash);
    expect(entry.license).toBe(s.license);
    expect(entry.license_url).toBe(s.licenseURL);
    expect(entry.metadata_status).toBe(s.status);
    expect(entry.derivation).toBe(s.derivation);
    expect(entry.provenance_confirmation).toBe('attention-translation-source-snapshot');
    if ('commit' in s) expect(entry.commit).toBe(s.commit);
    if ('evidence' in s) {
      expect(entry.license_evidence).toBe(s.evidence);
      expect(entry.license_evidence_sha256).toBe(s.evidenceHash);
      expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(s.evidenceHash);
    }
    if (s.kind === 'voita') expect(entry.rights_status).toBe(s.rights);
    if ('viewBox' in s) {
      expect(readFileSync(resolve(repositoryRoot, entry.asset), 'utf8')).toContain('viewBox="' + s.viewBox + '"');
    } else {
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual(s.size);
    }
    if ('crop' in s) {
      expect(entry.source_page).toBe(s.pageNumber);
      expect(entry.source_printed_page).toBe(s.printed);
      expect(entry.render_dpi).toBe(360);
      expect(entry.render_page_size_px).toEqual([2977,4210]);
      expect(entry.crop_box_px).toEqual(s.crop);
    }
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/seq2seq-bridge-2026-11-03/d2l-encoder-decoder.svg') {
    const expected = {
      "id": "curated-seq2seq-interface-bf5083530e10",
      "asset": "00 Учебник/Assets/Figures/curated/seq2seq-bridge-2026-11-03/d2l-encoder-decoder.svg",
      "author": "Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola",
      "title": "The encoder–decoder architecture, complete original D2L diagram",
      "source_url": "https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html",
      "source_image_url": "https://raw.githubusercontent.com/d2l-ai/d2l-en/23d7a5aecceee57d1292c56e90cce307f183bb0a/img/encoder-decoder.svg",
      "commit": "23d7a5aecceee57d1292c56e90cce307f183bb0a",
      "source_asset": "05 Источники/Originals/seq2seq-2026-10-21/encoder-decoder.svg",
      "source_asset_sha256": "bf5083530e10aff826bfc6be26b4b32ccd8ade3e7ad86cc7b7ee66dc5601ddaf",
      "sha256": "bf5083530e10aff826bfc6be26b4b32ccd8ade3e7ad86cc7b7ee66dc5601ddaf",
      "bytes": 22369,
      "source_viewbox": [
        0,
        0,
        304,
        34
      ],
      "source_page_snapshot": "05 Источники/Originals/seq2seq-2026-10-21/d2l-encoder-decoder.md",
      "source_page_sha256": "ad89663c337fc3aed42c26fb53365bb249a33ac96e45828adfb9ae0072cd0b05",
      "license": "CC BY-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
      "license_evidence": "site/public/licenses/d2l-23d7a5a-CC-BY-SA-4.0.txt",
      "license_evidence_sha256": "a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c",
      "public_license_copy": "site/public/licenses/d2l-23d7a5a-CC-BY-SA-4.0.txt",
      "derivation": "source-svg-copy",
      "metadata_status": "verified",
      "provenance_confirmation": "source-license-verified",
      "rights_status": "source-license-verified",
      "used_in": [
        "00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора.md"
      ]
    };
    for(const [field,value] of Object.entries(expected))
      expect((entry as unknown as Record<string,unknown>)[field],field).toEqual(value);
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot,entry.asset),'utf8')).toContain('viewBox="0 0 304 34"');
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (Object.hasOwn(seq2seqFigures, entry.asset)) {
    const s = seq2seqFigures[entry.asset as keyof typeof seq2seqFigures];
    const prefix = '05 Источники/Originals/seq2seq-2026-10-21/';
    expect(entry.id).toBe(s.id);
    expect(entry.author).toBe('Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola');
    expect(entry.commit).toBe(d2lCommit);
    expect(entry.source_url).toBe(s.url);
    expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/d2l-ai/d2l-en/' + d2lCommit + '/img/' + s.name + '.svg');
    expect(entry.source_asset).toBe(prefix + s.name + '.svg');
    expect(entry.sha256).toBe(s.hash);
    expect(entry.source_asset_sha256).toBe(s.hash);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(s.hash);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(s.hash);
    expect(entry.source_page_snapshot).toBe(prefix + 'd2l-' + s.page + '.md');
    expect(entry.source_page_sha256).toBe(s.pageHash);
    expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(s.pageHash);
    expect(entry.license).toBe('CC BY-SA 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-sa/4.0/');
    expect(entry.license_evidence).toBe('05 Источники/Originals/neural-language-model-2026-10-21/d2l-LICENSE.txt');
    expect(entry.license_evidence_sha256).toBe('a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.provenance_confirmation).toBe('seq2seq-source-snapshot');
    expect(entry.metadata_status).toBe('verified');
    expect(readFileSync(resolve(repositoryRoot, entry.asset), 'utf8')).toContain('viewBox="' + s.viewBox + '"');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/lstm-gru-contracts-2026-11-03/d2l-birnn.svg') {
    const expected = {
      "id": "curated-lstm-gru-bidirectional-aa4e678afa0e",
      "author": "Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola",
      "commit": "23d7a5aecceee57d1292c56e90cce307f183bb0a",
      "source_url": "https://d2l.ai/chapter_recurrent-modern/bi-rnn.html",
      "source_image_url": "https://raw.githubusercontent.com/d2l-ai/d2l-en/23d7a5aecceee57d1292c56e90cce307f183bb0a/img/birnn.svg",
      "source_asset": "05 Источники/Originals/lstm-gru-contracts-2026-11-03/d2l-birnn.svg",
      "source_asset_sha256": "aa4e678afa0e31c160374974709512d4d7e0121eeab7a1af1ad7725403afc018",
      "sha256": "aa4e678afa0e31c160374974709512d4d7e0121eeab7a1af1ad7725403afc018",
      "source_page_snapshot": "05 Источники/Originals/lstm-gru-contracts-2026-11-03/d2l-bi-rnn.md",
      "source_page_sha256": "6de8c8124497b0c69703ba3805559d51eb57c96d93096b70e2d5b6e133373975",
      "license": "CC BY-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
      "license_evidence": "site/public/licenses/d2l-23d7a5a-CC-BY-SA-4.0.txt",
      "license_evidence_sha256": "a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c",
      "public_license_copy": "site/public/licenses/d2l-23d7a5a-CC-BY-SA-4.0.txt",
      "derivation": "source-svg-copy",
      "metadata_status": "verified",
      "provenance_confirmation": "source-license-verified"
    };
    for (const [field,value] of Object.entries(expected))
      expect((entry as unknown as Record<string,unknown>)[field],field).toEqual(value);
    for (const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot,entry.asset),'utf8')).toContain('viewBox="0 0 240 141"');
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (Object.hasOwn(lstmGruFigures, entry.asset)) {
    const s = lstmGruFigures[entry.asset as keyof typeof lstmGruFigures];
    const prefix = '05 Источники/Originals/lstm-gru-2026-10-21/';
    expect(entry.provenance_confirmation).toBe('lstm-gru-source-snapshot');
    expect(entry.sha256).toBe(s.hash);
    expect(entry.source_asset_sha256).toBe(s.hash);
    expect(entry.source_asset).toBe(prefix + s.member);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(s.hash);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(s.hash);
    if (s.kind === 'd2l') {
      expect(entry.commit).toBe(d2lCommit);
      expect(entry.author).toBe('Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola');
      expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/d2l-ai/d2l-en/' + d2lCommit + '/img/' + s.sourceName + '.svg');
      expect(entry.source_url).toBe('https://d2l.ai/chapter_recurrent-modern/' + s.page + '.html');
      expect(entry.source_page_snapshot).toBe(prefix + 'd2l-' + s.page + '.md');
      expect(entry.source_page_sha256).toBe(s.page === 'lstm' ? '66a0dc190a4641d4e17bb14f8ad57cb066a46f1ee4f472a45a3138c588008614' : 'c7b6aaf75edac5e5d80dba8e77d482cf700b6ba3213427472871ea479c74465a');
      expect(entry.license).toBe('CC BY-SA 4.0');
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-sa/4.0/');
      expect(entry.license_evidence).toBe('05 Источники/Originals/neural-language-model-2026-10-21/d2l-LICENSE.txt');
      expect(entry.license_evidence_sha256).toBe('a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c');
      expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
      expect(entry.metadata_status).toBe('verified');
      expect(entry.derivation).toBe('source-svg-copy');
      expect(readFileSync(resolve(repositoryRoot, entry.asset), 'utf8')).toContain('viewBox="' + s.viewBox + '"');
    } else {
      expect(entry.author).toBe('Chris Olah');
      expect(entry.source_url).toBe('https://colah.github.io/posts/2015-08-Understanding-LSTMs/');
      expect(entry.source_image_url).toBe('https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/' + s.member);
      expect(entry.source_page_snapshot).toBe(prefix + 'colah-lstm.html');
      expect(entry.source_page_sha256).toBe('aa8f8269a2ee66fcf2ee0df23dbcd8f12c734238db5d4b676573acb01e83a868');
      expect(entry.license).toBe('User permission record (not a named license)');
      expect(entry.license_url).toBeNull();
      expect(entry.metadata_status).toBe('needs-confirmation');
      expect(entry.rights_status).toBe('source-verified; third-party-rights-not-independently-verified');
      expect(entry.derivation).toBe('source-raster-copy');
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([s.width, s.height]);
    }
    expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (Object.hasOwn(rnnBpttFigures, entry.asset)) {
    const specification = rnnBpttFigures[entry.asset as keyof typeof rnnBpttFigures];
    const prefix = '05 Источники/Originals/rnn-bptt-2026-10-21/';
    expect(entry.author).toBe('Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola');
    expect(entry.commit).toBe(d2lCommit);
    expect(entry.source_url).toBe('https://d2l.ai/chapter_recurrent-neural-networks/' + specification.page + '.html');
    expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/d2l-ai/d2l-en/' + d2lCommit + '/img/' + specification.name + '.svg');
    expect(entry.source_asset).toBe(prefix + 'd2l-' + specification.name + '.svg');
    expect(entry.sha256).toBe(specification.hash);
    expect(entry.source_asset_sha256).toBe(specification.hash);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(specification.hash);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(specification.hash);
    expect(entry.source_page_snapshot).toBe(prefix + 'd2l-' + specification.page + '.md');
    const pageHash = specification.page === 'rnn' ? 'cfa3068a1ff226b281c7743e7d852896486fcf797d4a9341f0dee70be92f161d' : 'fb24d5bc9a0f7a6847143f2e5af445bfa0acce2573a8d62bf981fc2986776448';
    expect(entry.source_page_sha256).toBe(pageHash);
    expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(pageHash);
    expect(entry.license).toBe('CC BY-SA 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-sa/4.0/');
    expect(entry.license_evidence).toBe('05 Источники/Originals/neural-language-model-2026-10-21/d2l-LICENSE.txt');
    expect(entry.license_evidence_sha256).toBe('a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.provenance_confirmation).toBe('rnn-bptt-source-snapshot');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('source-svg-copy');
    expect(readFileSync(resolve(repositoryRoot, entry.asset), 'utf8')).toContain('viewBox="' + specification.viewBox + '"');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }

  if (Object.hasOwn(wordEmbeddingFigures, entry.asset)) {
    const specification = wordEmbeddingFigures[entry.asset as keyof typeof wordEmbeddingFigures];
    const prefix = '05 Источники/Originals/word-embeddings-2026-10-21/';
    const kind = specification.kind;
    expect(entry.provenance_confirmation).toBe('word-embeddings-source-snapshot');
    expect(entry.sha256).toBe(specification.hash);
    expect(entry.source_asset_sha256).toBe(specification.hash);
    expect(entry.source_image_url).toBe(specification.url);
    expect(entry.source_asset).toBe(prefix + specification.member);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(specification.hash);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(specification.hash);
    expect(entry.author).toBe(kind === 'voita' ? 'Lena Voita' : kind === 'jay' ? 'Jay Alammar' : 'Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola');
    expect(entry.derivation).toBe(kind === 'd2l' ? 'source-svg-copy' : 'source-raster-copy');
    expect(entry.metadata_status).toBe(kind === 'voita' ? 'needs-confirmation' : 'verified');
    expect(entry.license).toBe(kind === 'voita' ? 'User permission record (not a named license)' : kind === 'jay' ? 'CC BY-NC-SA 4.0' : 'CC BY-SA 4.0');
    expect(entry.license_url).toBe(kind === 'voita' ? null : kind === 'jay' ? 'https://creativecommons.org/licenses/by-nc-sa/4.0/' : 'https://creativecommons.org/licenses/by-sa/4.0/');
    const page = kind === 'voita' ? 'voita-word-embeddings.html' : kind === 'jay' ? 'jay-illustrated-bert.html' : 'd2l-approx-training.md';
    const pageHash = kind === 'voita' ? '4f48bf06590077126727cca905cd503cc7dc3b9174f44072194ac09a37b2327e' : kind === 'jay' ? '7751808a235a98029be991c786d6218bbb91ee385a00dd857384f1ee9365432a' : 'ea2dd3bfa159d8f8bc8c57b426c7c2ba87def6f550ecec5af27d07a752e109ce';
    expect(entry.source_page_snapshot).toBe(prefix + page);
    expect(entry.source_page_sha256).toBe(pageHash);
    expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(pageHash);
    expect(entry.source_url).toBe(kind === 'voita' ? 'https://lena-voita.github.io/nlp_course/word_embeddings.html' : kind === 'jay' ? 'https://jalammar.github.io/illustrated-bert/#bert-for-feature-extraction' : 'https://d2l.ai/chapter_natural-language-processing-pretraining/approx-training.html#hierarchical-softmax');
    if (kind === 'voita') expect(entry.rights_status).toBe('source-verified; third-party-rights-not-independently-verified');
    if (kind === 'jay') expect(readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8')).toContain('creativecommons.org/licenses/by-nc-sa/4.0/');
    if (kind === 'd2l') {
      expect(entry.commit).toBe(d2lCommit);
      expect(entry.license_evidence).toBe('05 Источники/Originals/neural-language-model-2026-10-21/d2l-LICENSE.txt');
      expect(entry.license_evidence_sha256).toBe('a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c');
      expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
      expect(readFileSync(resolve(repositoryRoot, entry.asset), 'utf8')).toContain('viewBox="0 0 390 199"');
    } else {
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([specification.width, specification.height]);
    }
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }

  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/neural-language-model-2026-10-21/')) {
    const prefix = '05 Источники/Originals/neural-language-model-2026-10-21/';
    const hash = 'f426511da025523e7eac27ce91ccb4ef3b402fe0aad75bf0ce9207340bbfe547';
    expect(entry.asset).toBe('00 Учебник/Assets/Figures/curated/neural-language-model-2026-10-21/d2l-softmaxreg.svg');
    expect(entry.author).toBe('Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola');
    expect(entry.commit).toBe(d2lCommit);
    expect(entry.source_url).toBe('https://d2l.ai/chapter_linear-classification/softmax-regression.html#linear-model');
    expect(entry.source_image_url).toBe(`https://raw.githubusercontent.com/d2l-ai/d2l-en/${d2lCommit}/img/softmaxreg.svg`);
    expect(entry.source_asset).toBe(prefix + 'd2l-softmaxreg.svg');
    expect(entry.sha256).toBe(hash);
    expect(entry.source_asset_sha256).toBe(hash);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(hash);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(hash);
    expect(entry.source_page_snapshot).toBe(prefix + 'd2l-softmax.md');
    expect(entry.source_page_sha256).toBe('82c997368fa3003b2d8e44742da8e3bf4459ee7f1d24caba4e358f95166366b7');
    expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    expect(entry.license).toBe('CC BY-SA 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-sa/4.0/');
    expect(entry.license_evidence).toBe(prefix + 'd2l-LICENSE.txt');
    expect(entry.license_evidence_sha256).toBe('a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('neural-lm-source-snapshot');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    expect(readFileSync(resolve(repositoryRoot, entry.asset), 'utf8')).toContain('viewBox="0 0 263 88"');
    return;
  }

  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/voita-language-modeling/')) {
    const prefix = '05 Источники/Originals/ngram-2026-10-21/';
    const name = entry.asset.split('/').at(-1)!;
    const cases: Record<string, [string, string, number, number]> = {
      'ngram-window.png': ['bd5268c1c9371ce464870486b478c24b2672b918410c5406e99c2510f96fa44b', 'example_cut_3gram-min.png', 1877, 748],
      'ngram-probability.png': ['7e5287bdbb84ab8d31ade52b06e7eba9846c998f374e295f41c310a67748029a', 'prob_cat_on_a_mat-min.png', 1834, 196],
      'backoff.png': ['e9822923c163b072533515cf82cb0e899d7af0913c50449c4a0417a62d698b05', 'backoff-min.png', 982, 894],
      'interpolation.png': ['87ac856c87e342264e639ad4b133f53d5439502e2f558f9fff2e3140b067ec54', 'interpolation-min.png', 1174, 430],
    };
    expect(Object.hasOwn(cases, name)).toBe(true);
    const [hash, source, width, height] = cases[name]!;
    expect(entry.author).toBe('Lena Voita');
    expect(entry.source_url).toBe('https://lena-voita.github.io/nlp_course/language_modeling.html');
    expect(entry.source_image_url).toBe('https://lena-voita.github.io/resources/lectures/lang_models/ngram/' + source);
    expect(entry.source_asset).toBe(prefix + 'voita-' + name);
    expect(entry.sha256).toBe(hash);
    expect(entry.source_asset_sha256).toBe(hash);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(hash);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(hash);
    expect(entry.source_page_snapshot).toBe(prefix + 'voita-language-modeling.html');
    expect(entry.source_page_sha256).toBe('fab5bc7205c3a6a7694fd0d6b470c59888975dd0704124ffb9372c8798988b0d');
    expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    expect(entry.metadata_status).toBe('needs-confirmation');
    expect(entry.provenance_confirmation).toBe('user-confirmed-open-materials');
    expect(entry.rights_status).toBe('source-verified; third-party-rights-not-independently-verified');
    expect(entry.license).toBe('User permission record (not a named license)');
    expect(entry.license_url).toBeNull();
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([width, height]);
    return;
  }
  if (/\/voita-word-embeddings\/(?:one-hot\.png|lookup-table\.gif)$/.test(entry.asset)) {
    const prefix = '05 Источники/Originals/text-representation-2026-10-21/';
    const oneHot = entry.asset.endsWith('/one-hot.png');
    const hash = oneHot ? 'b1284449dfd787afc529b313bb3f95d08b5dfa8649034a57e370efc8a7239272' : '578a20d48490a7c42a4bb4c93cd7acfe0b5e96d3922c0384f452dee91708f0e0';
    expect(entry.author).toBe('Lena Voita');
    expect(entry.source_url).toBe('https://lena-voita.github.io/nlp_course/word_embeddings.html');
    expect(entry.source_image_url).toBe('https://lena-voita.github.io/resources/lectures/word_emb/' + (oneHot ? 'one_hot-min.png' : 'lookup_table.gif'));
    expect(entry.source_asset).toBe(prefix + (oneHot ? 'voita-one-hot.png' : 'voita-lookup-table.gif'));
    expect(entry.sha256).toBe(hash);
    expect(entry.source_asset_sha256).toBe(hash);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(hash);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(hash);
    expect(entry.source_page_snapshot).toBe(prefix + 'voita-word-embeddings.html');
    expect(entry.source_page_sha256).toBe('4f48bf06590077126727cca905cd503cc7dc3b9174f44072194ac09a37b2327e');
    expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    expect(entry.metadata_status).toBe('needs-confirmation');
    expect(entry.provenance_confirmation).toBe('user-confirmed-open-materials');
    expect(entry.rights_status).toBe('source-verified; third-party-rights-not-independently-verified');
    expect(entry.license).toBe('User permission record (not a named license)');
    expect(entry.license_url).toBeNull();
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    if (oneHot) expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([810, 802]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/initialization-normalization-2026-10-21/')) {
    const prefix = '05 Источники/Originals/initialization-normalization-2026-10-21/';
    const name = entry.asset.split('/').at(-1)!;
    const cases = {
      'd2l-residual-block.svg': ['357f558a8175536c522a7d5e9a1f49f29720694736c70b40ae99c31683ac901a', 'residual-block.svg'],
      'd2l-resnet-block.svg': ['b59554ec0382567909fc2575999cf91c745a62d7273cd9e0d4d088476f045e75', 'resnet-block.svg'],
      'cs336-prenorm-block.png': ['b0046068f94e2bb3ac56c4d6ba3001293649fc1a10bbf31fc12b1a607643029b', ''],
    };
    expect(Object.hasOwn(cases, name)).toBe(true);
    const [hash, member] = cases[name as keyof typeof cases];
    const d2l = name.startsWith('d2l');
    expect(entry.provenance_confirmation).toBe('norm-source-snapshot');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.sha256).toBe(hash);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(hash);
    expect(entry.author).toBe(d2l ? 'Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola' : 'Stanford CS336 course staff');
    expect(entry.license).toBe(d2l ? 'CC BY-SA 4.0' : 'MIT');
    expect(entry.license_url).toBe(d2l ? 'https://creativecommons.org/licenses/by-sa/4.0/' : 'https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/LICENSE');
    expect(entry.license_evidence).toBe(prefix + (d2l ? 'd2l-LICENSE.txt' : 'cs336-LICENSE.txt'));
    expect(entry.license_evidence_sha256).toBe(d2l ? 'a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c' : 'b3ccacaf9286910f22ba98681601f426288297f17004699f99d9fdd5f7432751');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.derivation).toBe(d2l ? 'source-svg-copy' : 'pdf-page-crop');
    expect(entry.source_asset).toBe(d2l ? prefix + name : '05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment1-basics/cs336_assignment1_basics.pdf');
    expect(entry.source_asset_sha256).toBe(d2l ? hash : 'cf402d1c6d3c66da035bdf33b3e2c1bbc4bb92a78b006ee36577e925a265e328');
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(entry.source_url).toBe(d2l ? 'https://d2l.ai/chapter_convolutional-modern/resnet.html#residual-blocks' : 'https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf');
    if (d2l) {
      expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/d2l-ai/d2l-en/' + d2lCommit + '/img/' + member);
      expect(entry.source_page_snapshot).toBe(prefix + 'd2l-resnet.md');
      expect(entry.source_page_sha256).toBe('64325be1880bac5ed388512412877b9fae653e4c967789c93319a4a91a316ed5');
      expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
      expect(readFileSync(resolve(repositoryRoot, entry.asset)).equals(readFileSync(resolve(repositoryRoot, entry.source_asset!)))).toBe(true);
    } else {
      expect(entry.source_page).toBe(13);
      expect(entry.render_dpi).toBe(300);
      expect(entry.render_page_size_px).toEqual([2550, 3300]);
      expect(entry.crop_box_px).toEqual([1390, 1710, 2220, 2850]);
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([830, 1140]);
      expect(readFileSync(resolve(repositoryRoot, 'site/public/licenses/stanford-cs336-assignment1-MIT.txt')).equals(readFileSync(resolve(repositoryRoot, entry.license_evidence!)))).toBe(true);
    }
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/optimization-2026-10-21/')) {
    const prefix = '05 Источники/Originals/optimization-2026-10-21/';
    const cases = [
      ['d2l-sgd-divergence.svg', 'd226ed64352db342b885e5d5c25fa9c34f6b1db88bd03fcfb9359edd8171a0d2', 'https://d2l.ai/_images/output_momentum_e3683f_15_1.svg'],
      ['d2l-momentum-weights.svg', 'e9e3fdd12dec59a0372fc68b9589e0bf3ddef55e55950b4325d83a7d17206280', 'https://d2l.ai/_images/output_momentum_e3683f_51_0.svg'],
      ['cs231n-nesterov.jpeg', 'af94b879e0913562974b00be9fdb1debcc177fa5595dd995f75d61e6c70e3a4b', 'https://raw.githubusercontent.com/cs231n/cs231n.github.io/8ec57ccfc32379ae4b37c45de949af359673f92f/assets/nn3/nesterov.jpeg'],
    ];
    const info = cases.find(c => entry.asset.endsWith('/' + c[0]));
    expect(info).toBeTruthy();
    const [name, hash, url] = info!;
    const d2l = name.startsWith('d2l');
    expect(entry.provenance_confirmation).toBe('optimization-source-snapshot');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.source_asset).toBe(prefix + name);
    expect(entry.sha256).toBe(hash);
    expect(entry.source_asset_sha256).toBe(hash);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(hash);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(hash);
    expect(entry.source_image_url).toBe(url);
    expect(entry.source_url).toBe(d2l ? 'https://d2l.ai/chapter_optimization/momentum.html' : url.replace('raw.githubusercontent.com/', 'github.com/').replace('/8ec57', '/blob/8ec57'));
    expect(entry.derivation).toBe(d2l ? 'source-svg-copy' : 'source-raster-copy');
    expect(entry.license).toBe(d2l ? 'CC BY-SA 4.0' : 'MIT');
    expect(entry.license_url).toBe(d2l ? 'https://creativecommons.org/licenses/by-sa/4.0/' : 'https://github.com/cs231n/cs231n.github.io/blob/8ec57ccfc32379ae4b37c45de949af359673f92f/LICENSE');
    expect(entry.license_evidence).toBe(prefix + (d2l ? 'd2l-LICENSE.txt' : 'cs231n-LICENSE.txt'));
    expect(entry.license_evidence_sha256).toBe(d2l ? 'a39a4b3d0f411a8217b7c6dd35dc8b06003784ce32d3593d0d2d780a7e9c2f8c' : '85b365f07b4cfc5d678c548bc41965bc3bc782106b8903bd3b876d9d49a4270d');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.source_page_snapshot).toBe(prefix + (d2l ? 'd2l-momentum.html' : 'cs231n-neural-networks-3.md'));
    expect(entry.source_page_sha256).toBe(d2l ? 'b92e018a6709e499a5f23fbcc81129d0525b3e0353caedd86761b40704803bc8' : '6aced5e0903fca06faeb263db9395a3373ec2d53680d859bdc96927c5f153d51');
    expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/micro-runtime-2026-10-27/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/micro-runtime-2026-10-27/assets.json'),'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(1);
    expect(entry).toEqual(rows[0]);
    expect(entry.commit).toBe('1cc91a97b40e1dddcecd3b0fbb30b6643e52f164');
    expect(entry.sha256).toBe('692475fcb0609be2a97f30078c3b9e84ac90130e14e2c705df483e922723ca6b');
    for(const [path,hash] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
    expect(entry.license).toBe('Apache-2.0');
    expect(entry.dimensions_px).toEqual([271,229]);
    expect(entry.derivation).toBe('source-raster-copy');
    expect(readFileSync(resolve(repositoryRoot,entry.license_evidence!)))
      .toEqual(readFileSync(resolve(repositoryRoot,'site/public/licenses/tflm-Apache-2.0.txt')));
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/training-step-2026-10-27/')) {
    const rows=JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/training-step-2026-10-27/assets.json'),'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(1);
    expect(entry).toEqual(rows[0]);
    expect(entry.commit).toBe('8ec57ccfc32379ae4b37c45de949af359673f92f');
    expect(entry.sha256).toBe('6d7bfc74b8315a0d7ab35d4cb8aeaae38bc439cc56be1c561b1b9a98b208fac9');
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot,entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot,entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    expect(sha256(resolve(repositoryRoot,entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.license).toBe('MIT');
    expect(entry.derivation).toBe('source-raster-copy');
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/backpropagation-2026-10-21/')) {
    const files = ['cs231n-chain.svg', 'cs231n-sigmoid.svg', 'cs231n-gates.svg'];
    const index = files.indexOf(entry.asset.split('/').at(-1)!);
    expect(index).toBeGreaterThanOrEqual(0);
    const commit = '8ec57ccfc32379ae4b37c45de949af359673f92f';
    expect(entry.commit).toBe(commit);
    expect(entry.source_url).toBe(`https://github.com/cs231n/cs231n.github.io/blob/${commit}/optimization-2.md`);
    expect(entry.source_asset).toBe('05 Источники/Originals/backpropagation-2026-10-21/cs231n-optimization-2.md');
    expect(entry.source_asset_sha256).toBe('f11216ba5f9611b039d11d614050c7ad7736df6b9f47ec83e5cfba2f336e366a');
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(entry.license).toBe('MIT');
    expect(entry.license_url).toBe(`https://github.com/cs231n/cs231n.github.io/blob/${commit}/LICENSE`);
    expect(entry.license_evidence).toBe('05 Источники/Originals/backpropagation-2026-10-21/cs231n-LICENSE.txt');
    expect(entry.license_evidence_sha256).toBe('85b365f07b4cfc5d678c548bc41965bc3bc782106b8903bd3b876d9d49a4270d');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(sha256(resolve(repositoryRoot, 'site/public/licenses/cs231n-MIT.txt'))).toBe(entry.license_evidence_sha256);
    expect(entry.provenance_confirmation).toBe('pinned-cs231n-inline-svg');
    expect(entry.derivation).toBe('source-inline-svg-export');
    expect(entry.metadata_status).toBe('verified');
    const source = readFileSync(resolve(repositoryRoot, entry.source_asset!), 'utf8');
    const svgs = source.match(/<svg\b[\s\S]*?<\/svg>/g)!;
    expect(svgs).toHaveLength(3);
    const expected = svgs[index].replace('viewbox=', 'viewBox=')
      .replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ') + '\n';
    expect(readFileSync(resolve(repositoryRoot, entry.asset), 'utf8')).toBe(expected);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/source-audit-1-18/gelu-figure-1.png') {
    const commit = '2726505c4767eed99dfd89ce0f836fbca44d6a6d';
    const base = 'https://github.com/hendrycks/GELUs/blob/' + commit;
    expect(entry.commit).toBe(commit);
    expect(entry.source_url).toBe(base + '/gelus.pdf#page=2');
    expect(entry.license).toBe('MIT');
    expect(entry.license_url).toBe(base + '/LICENSE');
    expect(entry.provenance_confirmation).toBe('activation-pdf-source-snapshot');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.source_asset).toBe('05 Источники/Originals/activation-functions-2026-10-21/gelus-author.pdf');
    expect(entry.source_asset_sha256).toBe('cebe02ef59174a386761849c09d66814332b9befb7dde28fdbdfdb72942427ec');
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(entry.license_evidence).toBe('05 Источники/Originals/activation-functions-2026-10-21/GELUs-LICENSE');
    expect(entry.license_evidence_sha256).toBe('d49e3ae394af6f81c519325fe6df22cdc8b2a4841b7085f960e607719f31ac3c');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(sha256(resolve(repositoryRoot, 'site/public/licenses/gelus-MIT.txt'))).toBe(entry.license_evidence_sha256);
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.source_page).toBe(2);
    expect(entry.render_dpi).toBe(240);
    expect(entry.crop_box_px).toEqual([912, 1040, 1696, 1632]);
    expect(entry.render_page_size_px).toEqual([2040, 2640]);
    expect(entry.sha256).toBe('940fbb31de0d2bdad800c8aaced3c39eeb8f15d50fed69bf6b2c702d6c3a4beb');
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/source-first-32-34/cs336-silu-vs-relu.png') {
    const commit = 'a158843b20107949f1a8d7df1b05cd33b9166712';
    const base = 'https://github.com/stanford-cs336/assignment1-basics/blob/' + commit;
    const original = '05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment1-basics/';
    expect(entry.commit).toBe(commit);
    expect(entry.source_url).toBe(base + '/cs336_assignment1_basics.pdf#page=21');
    expect(entry.license).toBe('MIT');
    expect(entry.license_url).toBe(base + '/LICENSE');
    expect(entry.provenance_confirmation).toBe('activation-pdf-source-snapshot');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.source_asset).toBe(original + 'cs336_assignment1_basics.pdf');
    expect(entry.source_asset_sha256).toBe('cf402d1c6d3c66da035bdf33b3e2c1bbc4bb92a78b006ee36577e925a265e328');
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(entry.license_evidence).toBe(original + 'LICENSE');
    expect(entry.license_evidence_sha256).toBe('b3ccacaf9286910f22ba98681601f426288297f17004699f99d9fdd5f7432751');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(sha256(resolve(repositoryRoot, 'site/public/licenses/stanford-cs336-assignment1-MIT.txt'))).toBe(entry.license_evidence_sha256);
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.source_page).toBe(21);
    expect(entry.render_dpi).toBe(240);
    expect(entry.crop_box_px).toEqual([633, 293, 1420, 1084]);
    expect(entry.render_page_size_px).toEqual([2040, 2640]);
    expect(entry.sha256).toBe('59a942959793b641878bf65c6e83c6941e5153dc09a9893c6a5cb44f279f70e0');
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    return;
  }
  if (entry.provenance_confirmation === 'commons-lloyd-source-snapshot') {
    const cases = [
      [1,"5/5e","448124955","f31a1adf9aa5db300fab6959a14ad291e6177c21c0276c1b3d1d8e5a77cc6b4b","34385023a9ea347b7e1a623be676350a537344fed904de004ae83e89a0a4d3cb","197.17","190"],
      [2,"a/a5","1216685096","b90fd12be5a428a344a7d91732876d40f86ebed9d0d017cb4c40fe4d22f36f54","07e0d485f26da349a52e2b5c42be711a12a6e18735ab2a288da310216dbd4839","197.17","170"],
      [3,"3/3e","1219119226","a6e8c4d05c433fa6ef3b1909ea9412485819930056ccf6e40c13cf95f89bf45d","6f380d71bdd7bf2b99badeac1d9b40ee1b5295479f093d40ae61ca7365b7d766","197.14","170"],
      [4,"d/d2","1174293830","4cb2947c6be9f75aded27479cd6033588dbdf3f781723ce30165a1eca94254e3","9b763ea61866728abd8bfdd97bebf8b0a723c88ddd9d4838deb2dd7c91657330","197.17","170"]
    ] as const;
    const prefix = '00 Учебник/Assets/Figures/curated/classical-ml-clustering/';
    const originals = '05 Источники/Originals/clustering-2026-10-22/';
    const expected = cases.find(([step]) => entry.asset === prefix + 'lloyd-step-' + step + '.svg');
    expect(expected).toBeDefined();
    const [step, dir, oldid, hash, pageHash, width, height] = expected!;
    const name = 'K_Means_Example_Step_' + step + '.svg';
    const page = originals + 'step-' + step + '-page.html';
    expect(entry.author).toBe('Weston.pace');
    expect(entry.source_url).toBe('https://commons.wikimedia.org/w/index.php?title=File:' + name + '&oldid=' + oldid);
    expect(entry.source_image_url).toBe('https://upload.wikimedia.org/wikipedia/commons/' + dir + '/' + name);
    expect(entry.source_asset).toBe(originals + name);
    expect(entry.source_asset_sha256).toBe(hash);
    expect(entry.sha256).toBe(hash);
    expect(entry.source_page_snapshot).toBe(page);
    expect(entry.source_page_sha256).toBe(pageHash);
    expect(entry.license).toBe('CC BY-SA 3.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-sa/3.0/');
    expect(entry.license_evidence).toBe(page);
    expect(entry.license_evidence_sha256).toBe(pageHash);
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.metadata_status).toBe('verified');
    expect(entry.modifications).toContain('byte-for-byte');
    for (const [path, digest] of [[entry.asset, hash], [entry.source_asset, hash], [page, pageHash]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot, entry.asset)).equals(readFileSync(resolve(repositoryRoot, entry.source_asset!)))).toBe(true);
    const evidence = readFileSync(resolve(repositoryRoot, page), 'utf8');
    for (const token of ['Weston.pace', name, 'creativecommons.org/licenses/by-sa/3.0/'])
      expect(evidence).toContain(token);
    const svg = readFileSync(resolve(repositoryRoot, entry.asset), 'utf8');
    expect(svg).toContain('<svg');
    expect(svg).toContain('width="' + width + '"');
    expect(svg).toContain('height="' + height + '"');
    expect(svg.match(/<rect\b/g)).toHaveLength(12);
    expect(svg).not.toMatch(/<script\b|<foreignObject\b|on(?:load|error)\s*=/i);
    return;
  }
  const clusteringCourseOutput = [
    ['mooc-news-external-metrics.png', '712206c5a2610cac4dc2010e563719e91af6d217784fea59941e2a53a4f16f1b', 'clustering_supervised_metrics', 611, 879],
    ['mooc-news-silhouette.png', '5604f75c9275f74e35b7679a4980af1f46372513ede60f273859d48cddcbca84', 'clustering_supervised_metrics', 615, 491],
    ['mooc-rfm-quantile-layers.png', 'cfa0c2204e8cb75cd4039771c841ab49d814bc10bd48fc555ca77eeefad574ef', 'clustering_sol_01', 766, 766],
  ].find(([name]) => entry.asset === '00 Учебник/Assets/Figures/curated/classical-ml-clustering/' + name);
  if (clusteringCourseOutput) {
    const [, hash, page, width, height] = clusteringCourseOutput;
    const pin = '93840904e26a215ecddf1a30e9a5eb920284fb29';
    expect(entry.sha256).toBe(hash);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(hash);
    expect(entry.commit).toBe(pin);
    expect(entry.source_url).toBe('https://github.com/INRIA/scikit-learn-mooc/blob/' + pin + '/python_scripts/' + page + '.html');
    expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/' + pin + '/_images/' + hash + '.png');
    expect(entry.author).toBe('INRIA, scikit-learn MOOC contributors');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE');
    expect(entry.license_evidence).toBe('05 Источники/Originals/train-validation-test-2026-10-21/mooc-LICENSE.txt');
    expect(entry.license_evidence_sha256).toBe('f5b745ef98087f531e719ee8ca6a96809444573ecc7173c6fa68eaad39b3cc3f');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.crop_box_px).toBeUndefined();
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([width, height]);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/classical-ml-clustering/')) {
    const cases = [["elbow.png","e80edba3e4de8072dc41b696461815d7c4a531a7bf119b4488dbd6efbdc553e9","kmeans",""],["silhouette.png","d2b3f06031a118d871972f35e3d65dfbd06edaf47a90bc71aa89ed60aee3b8de","kmeans",""],["shapes-kmeans.png","b0e37de75e80cd7cab5dfc9981f8490ec2b77d61ca268377322cccd4444fbb4f","density",""],["shapes-hdbscan.png","21622047b9935ff74be9967cad95862a43e1272044fa613d765dbcac493358d4","density",""],["dbscan-core-border.png","6fbc56c13ca9ce3e0ef4632374378a2417fa86f2891c119ddb41abe27c7070cd","dbscan","sphx_glr_plot_dbscan_002.png"],["hdbscan-mst.png","769261ea803560add0ee71441b4085fde4d86e2d5c40922d64a8b1873aa789a6","hdb","how_hdbscan_works_10_1.png"],["hdbscan-condensed-selected.png","65613bb03af4ddfedaa948d2ab12df7af72d2a7117fe72bf136ab554b6e57a6d","hdb","how_hdbscan_works_18_1.png"],["hdbscan-mutual-reachability.svg","c6b8a7267e7e5a2346b2f0167ed7c3ae9234df6575e7aa483f007d5691d5dca5","hdb","distance4a.svg"]] as const;
    const groups = {"kmeans":{"url":"https://inria.github.io/scikit-learn-mooc/python_scripts/clustering_kmeans.html","page":"05 Источники/Originals/clustering-2026-10-21/kmeans-page.html","hash":"4fd9942bb6f2c21ec2b31fd017829199228b76addc3cbfde6c6c82bef1b9e224"},"density":{"url":"https://inria.github.io/scikit-learn-mooc/python_scripts/clustering_hdbscan.html","page":"05 Источники/Originals/clustering-2026-10-21/density-page.html","hash":"98ee391e575fea90a90b63592ba7bd4172e25dc890dae1b262ea57b3c11ebbc6"},"dbscan":{"url":"https://scikit-learn.org/stable/modules/clustering.html#dbscan","page":"05 Источники/Originals/clustering-2026-10-21/sklearn-guide.html","hash":"3c8d5481d8092e86fe47ff269e134510dedd006360aeaf30b9b982c91a01de1c"},"hdb":{"url":"https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html","page":"05 Источники/Originals/clustering-2026-10-21/hdbscan-how.rst","hash":"ddd3ac9417127b4509844b8ff4e967259f60802ed29d4a5d3a145724ee0d8f26"}} as const;
    const found = cases.find(row => entry.asset === '00 Учебник/Assets/Figures/curated/classical-ml-clustering/' + row[0]);
    expect(found).toBeDefined();
    const [name, hash, key, sourceName] = found!;
    const mooc = key === 'kmeans' || key === 'density';
    const group = groups[key];
    expect(entry.provenance_confirmation).toBe('clustering-source-snapshot');
    expect(entry.author).toBe(mooc ? 'scikit-learn MOOC authors' : key === 'hdb' ? 'Leland McInnes' : 'The scikit-learn developers');
    expect(entry.sha256).toBe(hash);
    expect(entry.source_asset).toBe('05 Источники/Originals/clustering-2026-10-21/' + name);
    expect(entry.source_asset_sha256).toBe(hash);
    expect(entry.source_url).toBe(group.url);
    expect(entry.source_image_url).toBe(mooc ? 'https://inria.github.io/scikit-learn-mooc/_images/' + hash + '.png'
      : key === 'hdb' ? 'https://raw.githubusercontent.com/scikit-learn-contrib/hdbscan/bb01526bf18c92e35ca0b4f5307c6d4248ddf5dc/docs/images/' + sourceName
      : 'https://scikit-learn.org/stable/_images/' + sourceName);
    expect(entry.source_page_snapshot).toBe(group.page);
    expect(entry.source_page_sha256).toBe(group.hash);
    expect(entry.license).toBe(mooc ? 'CC BY 4.0' : 'BSD-3-Clause');
    expect(entry.license_url).toBe(mooc
      ? 'https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE'
      : key === 'hdb' ? 'https://github.com/scikit-learn-contrib/hdbscan/blob/bb01526bf18c92e35ca0b4f5307c6d4248ddf5dc/LICENSE'
      : 'https://github.com/scikit-learn/scikit-learn/blob/1.9.1/COPYING');
    expect(entry.license_evidence).toBe(mooc
      ? '05 Источники/Originals/train-validation-test-2026-10-21/mooc-LICENSE.txt'
      : '05 Источники/Originals/clustering-2026-10-21/' + (key === 'hdb' ? 'hdbscan-LICENSE' : 'sklearn-COPYING'));
    expect(entry.license_evidence_sha256).toBe(mooc
      ? 'f5b745ef98087f531e719ee8ca6a96809444573ecc7173c6fa68eaad39b3cc3f'
      : key === 'hdb' ? 'd5be17fef04ce2213c09a3fffec11021065eaff565733c46ef287248b353a081'
      : '50d6a9d340f19ab355609917993114daf5f47e3161067bcf34955bbd05cd9cb0');
    for (const [file, digest] of [[entry.asset, hash], [entry.source_asset, hash],
      [group.page, group.hash], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot, entry.asset)).equals(readFileSync(resolve(repositoryRoot, entry.source_asset!)))).toBe(true);
    expect(readFileSync(resolve(repositoryRoot, group.page), 'utf8')).toContain(mooc ? hash + '.png' : sourceName);
    expect(entry.derivation).toBe(name.endsWith('.svg') ? 'source-svg-copy' : 'source-raster-copy');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.metadata_status).toBe('verified');
    if (key === 'hdb') {
      expect(entry.commit).toBe('bb01526bf18c92e35ca0b4f5307c6d4248ddf5dc');
      expect(readFileSync(resolve(repositoryRoot, 'site/public/licenses/hdbscan-BSD-3-Clause.txt')).equals(readFileSync(resolve(repositoryRoot, entry.license_evidence!)))).toBe(true);
    }
    if (key === 'dbscan') {
      expect(entry.source_code).toBe('05 Источники/Originals/clustering-2026-10-21/dbscan-code.py');
      expect(entry.source_code_sha256).toBe('a1a3f32b6fa917914ae103f9af125ddb4bf9140e6e3ff8189c69d8a2d6f63460');
      expect(sha256(resolve(repositoryRoot, entry.source_code!))).toBe(entry.source_code_sha256);
      expect(readFileSync(resolve(repositoryRoot, entry.source_code!), 'utf8')).toContain('SPDX-License-Identifier: BSD-3-Clause');
    }
    if (name.endsWith('.png')) expectDecodableImage(resolve(repositoryRoot, entry.asset));
    else expect(readFileSync(resolve(repositoryRoot, entry.asset), 'utf8')).toContain('<svg');
    return;
  }
  if (entry.provenance_confirmation === 'cv-indices-build-snapshot') {
    const prefix = '00 Учебник/Assets/Figures/curated/classical-ml-evaluation/';
    const originals = '05 Источники/Originals/evaluation-2026-10-22/';
    const cases = [
      ['group-kfold.png', '004', '8a4680c6a641d4d7f29025a131dd344b11e017ac09553028a3819e6fbbb9f6cb'],
      ['time-series-split.png', '013', '3e57bcd142a9cd0a6e32ca9930358dd56a42952a93bf3c8b1d46a979a4c8df08']
    ];
    const expected = cases.find(([name]) => entry.asset === prefix + name);
    expect(expected).toBeDefined();
    const [name, imageIndex, hash] = expected!;
    expect(entry.author).toBe('The scikit-learn developers');
    expect(entry.source_url).toBe('https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html');
    const imageName = 'sphx_glr_plot_cv_indices_' + imageIndex + '.png';
    expect(entry.source_image_url).toBe('https://scikit-learn.org/stable/_images/' + imageName);
    expect(entry.source_asset).toBe(originals + name);
    expect(entry.source_asset_sha256).toBe(hash);
    expect(entry.sha256).toBe(hash);
    expect(entry.source_page_snapshot).toBe(originals + 'cv-indices-page.html');
    expect(entry.source_page_sha256).toBe('3319d4b5b4fa46fbba76773658b0b18d0befec06168536f3b28790ad1644a83f');
    expect(entry.source_code).toBe(originals + 'plot_cv_indices.py');
    expect(entry.source_code_sha256).toBe('56315d014009c15000559e8161b9c88c43c53022991efa8dbf74bad54b4d9b28');
    expect(entry.commit).toBe('866c0f51e7560ef0303cbcc5f159df5382ea9e3f');
    expect(entry.source_code_url).toBe('https://github.com/scikit-learn/scikit-learn/blob/' + entry.commit + '/examples/model_selection/plot_cv_indices.py');
    expect(entry.license).toBe('BSD-3-Clause');
    expect(entry.license_url).toBe('https://github.com/scikit-learn/scikit-learn/blob/1.9.1/COPYING');
    expect(entry.license_evidence).toBe('05 Источники/Originals/evaluation-2026-10-21/sklearn-COPYING');
    expect(entry.license_evidence_sha256).toBe('50d6a9d340f19ab355609917993114daf5f47e3161067bcf34955bbd05cd9cb0');
    for (const [path, digest] of [[entry.asset, hash], [entry.source_asset, hash],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.source_code, entry.source_code_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot, entry.asset)).equals(readFileSync(resolve(repositoryRoot, entry.source_asset!)))).toBe(true);
    expect(readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8')).toContain(imageName);
    expect(readFileSync(resolve(repositoryRoot, entry.source_code!), 'utf8')).toContain('SPDX-License-Identifier: BSD-3-Clause');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.metadata_status).toBe('verified');
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([600, 300]);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/classical-ml-evaluation/regression-transformed-target.png') {
    const hash = '42c821c57965432baa2d5f05d48516ebf94f928f0fabdc42497b7dc019373d9d';
    const pin = '93840904e26a215ecddf1a30e9a5eb920284fb29';
    expect(entry.sha256).toBe(hash);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(hash);
    expect(entry.commit).toBe(pin);
    expect(entry.source_url).toBe('https://github.com/INRIA/scikit-learn-mooc/blob/' + pin + '/python_scripts/metrics_regression.html');
    expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/' + pin + '/_images/' + hash + '.png');
    expect(entry.author).toBe('INRIA, scikit-learn MOOC contributors');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE');
    expect(entry.license_evidence).toBe('05 Источники/Originals/train-validation-test-2026-10-21/mooc-LICENSE.txt');
    expect(entry.license_evidence_sha256).toBe('f5b745ef98087f531e719ee8ca6a96809444573ecc7173c6fa68eaad39b3cc3f');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.crop_box_px).toBeUndefined();
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([1079, 585]);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/classical-ml-evaluation/')) {
    const cases = [["pr-curve.png","5fe0c4d7c02afccc1426b309935973ee3b9484e057699ebcee96a600cf512fa1","pr",[0,68,643,694]],["roc-curve.png","8795e71f48bdc87fb880ab0637ea6982bf7020a14a310aaa47cf86dd57471b65","pr",[645,68,1280,694]],["pr-roc.png","95bc5b5092ec82cf470d26f1c850f887b8ac5cccb41370d4ef4f487ba4c7284a","pr",null],["calibration-curve.png","567820d0a5ddf1ca4369b6a4332eb6e775a42682e49a0343128439d811beb8f8","cal",[0,0,1000,500]],["calibration.png","169a45a50ea9f78c948cfbd37977c7bd55fd507d88ff8141c9a9c7d272d07502","cal",null],["regression-actual.png","2e7e87689175b0876e7e6673b8d4bb47e8b1c432c8e74bcdd3449db7d9b964af","reg",[0,100,540,585]],["regression-residual.png","abc859c1cdd488c2c5ee1d69e305214dca998563691305997caf4690c231722f","reg",[550,100,1069,585]],["regression-residuals.png","5c1783874df4d41f5ad0239786afd5b680b318058dc4a00ed7f9c24e4d555e33","reg",null],["nested-first-outer.png","5373ad8243c69055ee823ce4625f6839a7899ea68ae977b0761ff281c4b755fa","nested",[0,95,915,370]],["nested-legend.png","7a024a609461d5ac8710252d5da018ddc005bf2f33ba8ee0253007e7d4bbd982","nested",[960,95,1310,355]],["calibration-legend.png","f6a6179303993eefb7161499eb3a2a26505565560054f17acf31cd9f5458e451","cal",[785,325,985,447]]] as const;
    const groups = {"pr":{"file":"05 Источники/Originals/evaluation-2026-10-21/pr-roc.png","hash":"95bc5b5092ec82cf470d26f1c850f887b8ac5cccb41370d4ef4f487ba4c7284a","url":"https://inria.github.io/scikit-learn-mooc/python_scripts/metrics_classification.html","image":"https://inria.github.io/scikit-learn-mooc/_images/95bc5b5092ec82cf470d26f1c850f887b8ac5cccb41370d4ef4f487ba4c7284a.png","page":"05 Источники/Originals/evaluation-2026-10-21/classification-page.html","pageHash":"36b93ccfcd500a583a67d2bcbe8e5d1e6a90de2e775ba8fc3d8d568049bdcb50"},"reg":{"file":"05 Источники/Originals/evaluation-2026-10-21/regression-residuals.png","hash":"5c1783874df4d41f5ad0239786afd5b680b318058dc4a00ed7f9c24e4d555e33","url":"https://inria.github.io/scikit-learn-mooc/python_scripts/metrics_regression.html","image":"https://inria.github.io/scikit-learn-mooc/_images/5c1783874df4d41f5ad0239786afd5b680b318058dc4a00ed7f9c24e4d555e33.png","page":"05 Источники/Originals/evaluation-2026-10-21/regression-page.html","pageHash":"f10b18a75b83573e1080cfc0acf44ee9adfcf97cf97c5dc885f91a0e9f0c42e3"},"cal":{"file":"05 Источники/Originals/evaluation-2026-10-21/calibration.png","hash":"169a45a50ea9f78c948cfbd37977c7bd55fd507d88ff8141c9a9c7d272d07502","url":"https://scikit-learn.org/stable/modules/calibration.html","image":"https://scikit-learn.org/stable/_images/sphx_glr_plot_compare_calibration_001.png","page":"05 Источники/Originals/evaluation-2026-10-21/calibration-page.html","pageHash":"00860ac6c19e425758b76adf27721d343ad3d5cbc72f865da7a9fe1aa32b9864"},"nested":{"file":"Assets/Sources/Scikit-learn MOOC/figures/nested_cross_validation_diagram.png","hash":"2aa238af6085b56c531e8a687cfa4f4ebe0c91bc89ef9f667475f7c371d5b842","url":"https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/figures/nested_cross_validation_diagram.png","image":"https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/figures/nested_cross_validation_diagram.png"}} as const;
    const found = cases.find(row => entry.asset.endsWith('/' + row[0]));
    expect(found).toBeDefined();
    const [name, hash, key, crop] = found!;
    const source = groups[key];
    expect(entry.provenance_confirmation).toBe('evaluation-source-snapshot');
    expect(entry.sha256).toBe(hash);
    expect(entry.source_asset).toBe(source.file);
    expect(entry.source_asset_sha256).toBe(source.hash);
    expect(entry.source_url).toBe(source.url);
    expect(entry.source_image_url).toBe(source.image);
    expect(entry.author).toBe(key === 'cal' ? 'The scikit-learn developers' : 'scikit-learn MOOC authors');
    expect(entry.license).toBe(key === 'cal' ? 'BSD-3-Clause' : 'CC BY 4.0');
    expect(entry.license_url).toBe(key === 'cal'
      ? 'https://github.com/scikit-learn/scikit-learn/blob/1.9.1/COPYING'
      : 'https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE');
    expect(entry.license_evidence).toBe(key === 'cal'
      ? '05 Источники/Originals/evaluation-2026-10-21/sklearn-COPYING'
      : '05 Источники/Originals/train-validation-test-2026-10-21/mooc-LICENSE.txt');
    expect(entry.license_evidence_sha256).toBe(key === 'cal'
      ? '50d6a9d340f19ab355609917993114daf5f47e3161067bcf34955bbd05cd9cb0'
      : 'f5b745ef98087f531e719ee8ca6a96809444573ecc7173c6fa68eaad39b3cc3f');
    for (const [path, digest] of [[entry.asset, hash], [source.file, source.hash],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(digest);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe(crop ? 'source-raster-crop' : 'source-raster-copy');
    if (crop) {
      expect(entry.crop_box_px).toEqual(crop);
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([crop[2]-crop[0], crop[3]-crop[1]]);
    } else {
      expect(entry.crop_box_px).toBeUndefined();
      expect(readFileSync(resolve(repositoryRoot, entry.asset)).equals(readFileSync(resolve(repositoryRoot, source.file)))).toBe(true);
    }
    if ('page' in source) {
      expect(entry.source_page_snapshot).toBe(source.page);
      expect(entry.source_page_sha256).toBe(source.pageHash);
      expect(sha256(resolve(repositoryRoot, source.page))).toBe(source.pageHash);
      expect(readFileSync(resolve(repositoryRoot, source.page), 'utf8')).toContain(source.image.split('/').at(-1));
    }
    if (key === 'cal') {
      expect(entry.source_code).toBe('05 Источники/Originals/evaluation-2026-10-21/calibration-code.py');
      expect(entry.source_code_sha256).toBe('1d3ed3e8e072e4a12e381927514b40944357e83c79dc2e0799be95b6460e309f');
      expect(sha256(resolve(repositoryRoot, entry.source_code!))).toBe(entry.source_code_sha256);
      expect(readFileSync(resolve(repositoryRoot, entry.source_code!), 'utf8')).toContain('SPDX-License-Identifier: BSD-3-Clause');
    }
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.provenance_confirmation === 'official-mooc-page-snapshot'
    && entry.asset.startsWith('00 Учебник/Assets/Figures/curated/classical-ml-ensembles/')) {
    const originals = '05 Источники/Originals/ensembles-2026-10-21/';
    const cases = [["bagging-hundred-average.png","71b3ddc7bcc100af144a93c00a8e87a7f665b9dcd4532b73c3c42793107135c1","bagging-hundred-average.png","71b3ddc7bcc100af144a93c00a8e87a7f665b9dcd4532b73c3c42793107135c1","bagging",null],["bagging-three-average.png","935b6fa4b6fb39d7827b96ecffe783e9b03653c486794aa1470830b955897b8d","bagging-three-average.png","935b6fa4b6fb39d7827b96ecffe783e9b03653c486794aa1470830b955897b8d","bagging",null],["bagging-three-trees.png","aa2883cee12aed2a188a999cf379dc876ac90dad773b9870ca12f448cc464129","bagging-three-trees.png","aa2883cee12aed2a188a999cf379dc876ac90dad773b9870ca12f448cc464129","bagging",null],["bootstrap-sample.png","de65309953f108ee9132d1c3326a630102d4d91237c24de9010b6a5a375ada3f","bootstrap-sample.png","de65309953f108ee9132d1c3326a630102d4d91237c24de9010b6a5a375ada3f","bagging",null],["gradient-fit-residuals.png","a4a3778869152286bcd5c54c53631d208f0896259b4486841ded4f0f91dfaff2","gradient-fit-residuals.png","a4a3778869152286bcd5c54c53631d208f0896259b4486841ded4f0f91dfaff2","gradient",null],["gradient-initial-legend.png","ef42284c96b1ce1f547a9d1a28879f6de4f56dfc53625f9e9c724dd9d02cbef7","gradient-initial-residuals.png","9a116965bb8133ee4bfe4d8d67625b2e52709f10e94cf871655ba041d8f50cd5","gradient",[615,145,934,240]],["gradient-initial-plot.png","f52097fa8531817dc25e009c14a94ec19ecd0c5d9ba26024d94279d336b615d0","gradient-initial-residuals.png","9a116965bb8133ee4bfe4d8d67625b2e52709f10e94cf871655ba041d8f50cd5","gradient",[0,0,610,520]],["gradient-initial-residuals.png","9a116965bb8133ee4bfe4d8d67625b2e52709f10e94cf871655ba041d8f50cd5","gradient-initial-residuals.png","9a116965bb8133ee4bfe4d8d67625b2e52709f10e94cf871655ba041d8f50cd5","gradient",null],["gradient-residual-legend.png","f3a8e811ef6cb09f5e08a19bd456538b99105da8987d36facb67ac3641fe1de5","gradient-fit-residuals.png","a4a3778869152286bcd5c54c53631d208f0896259b4486841ded4f0f91dfaff2","gradient",[640,112,957,240]],["gradient-residual-plot.png","5a3fe3b249365d8aa5283676cd151a5268eb3df6feddacb99f0e220166fc757b","gradient-fit-residuals.png","a4a3778869152286bcd5c54c53631d208f0896259b4486841ded4f0f91dfaff2","gradient",[0,0,635,487]]] as const;
    const expected = cases.find(row => entry.asset.endsWith('/' + row[0]));
    expect(expected).toBeDefined();
    const [name, hash, source, sourceHash, page, crop] = expected!;
    expect(entry.source_url).toBe('https://inria.github.io/scikit-learn-mooc/python_scripts/ensemble_' + (page === 'bagging' ? 'bagging' : 'gradient_boosting') + '.html');
    expect(entry.source_image_url).toBe('https://inria.github.io/scikit-learn-mooc/_images/' + sourceHash + '.png');
    expect(entry.source_asset).toBe(originals + source);
    expect(entry.source_asset_sha256).toBe(sourceHash);
    expect(entry.sha256).toBe(hash);
    expect(entry.source_page_snapshot).toBe(originals + page + '-page.html');
    expect(entry.source_page_sha256).toBe(page === 'bagging'
      ? '8f045ece78f865710eb6eb7c79c144e2ae169a21465a2bf550c29680aaf76ef4'
      : '785dbfcbc70283968f19407a96dfac8a6422105215405d5c25016e996b13e6b5');
    expect(entry.author).toBe('scikit-learn MOOC authors');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE');
    expect(entry.license_evidence).toBe('05 Источники/Originals/train-validation-test-2026-10-21/mooc-LICENSE.txt');
    expect(entry.license_evidence_sha256).toBe('f5b745ef98087f531e719ee8ca6a96809444573ecc7173c6fa68eaad39b3cc3f');
    for (const [file, digest] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8')).toContain(sourceHash + '.png');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe(crop ? 'source-raster-crop' : 'source-raster-copy');
    if (crop) {
      expect(entry.crop_box_px).toEqual(crop);
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([crop[2]-crop[0], crop[3]-crop[1]]);
    } else {
      expect(entry.crop_box_px).toBeUndefined();
      expect(readFileSync(resolve(repositoryRoot, entry.asset)).equals(readFileSync(resolve(repositoryRoot, entry.source_asset!)))).toBe(true);
    }
    return;
  }
  if (entry.provenance_confirmation === 'official-mooc-page-snapshot') {
    const prefix = '00 Учебник/Assets/Figures/curated/classical-ml-trees/';
    const originals = '05 Источники/Originals/decision-trees-2026-10-21/';
    const cases = [
      ['penguins-tree.png', '8a01523b117bd8c902e55b9f615a5743fbd1630430dfe3d81da24d8423454bc7'],
      ['penguins-probabilities.png', 'ef3d0aafee83c1fa35c85b5965f3df8b3411686652a99213e2168e624dba2fef']
    ];
    const expected = cases.find(([name]) => entry.asset === prefix + name);
    expect(expected).toBeDefined();
    const [name, hash] = expected!;
    expect(entry.author).toBe('scikit-learn MOOC authors');
    expect(entry.source_url).toBe('https://inria.github.io/scikit-learn-mooc/python_scripts/trees_classification.html');
    expect(entry.source_image_url).toBe('https://inria.github.io/scikit-learn-mooc/_images/' + hash + '.png');
    expect(entry.source_asset).toBe(originals + name);
    expect(entry.sha256).toBe(hash);
    expect(entry.source_asset_sha256).toBe(hash);
    expect(entry.source_page_snapshot).toBe(originals + 'classification-page.html');
    expect(entry.source_page_sha256).toBe('74994adaa33ee64dd2ca6a7ca6961560b3bd9edc73cdf8a5c5ae6721aea2ea40');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE');
    expect(entry.license_evidence).toBe('05 Источники/Originals/train-validation-test-2026-10-21/mooc-LICENSE.txt');
    expect(entry.license_evidence_sha256).toBe('f5b745ef98087f531e719ee8ca6a96809444573ecc7173c6fa68eaad39b3cc3f');
    for (const [file, digest] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot, entry.asset)).equals(readFileSync(resolve(repositoryRoot, entry.source_asset!)))).toBe(true);
    expect(readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8')).toContain(hash + '.png');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.crop_box_px).toBeUndefined();
    return;
  }
  if (entry.provenance_confirmation === 'official-sklearn-gallery-snapshot') {
    const originals = '05 Источники/Originals/linear-models-2026-10-21/';
    const names = ['sklearn-polynomial-spline-basis.png', 'sklearn-polynomial-basis.png', 'sklearn-spline-basis.png'];
    const index = names.findIndex(name => entry.asset === '00 Учебник/Assets/Figures/curated/classical-ml-linear/' + name);
    expect(index).toBeGreaterThanOrEqual(0);
    const hashes = ['a5c8285db6267f4a931501c52f16f95dfb8086e5a6f938c7b5ef3889ff012a25',
      '6b956432b7d465d6e2a4823e3554a2355e70058256dda9489019c4e8314de377',
      '047d9c96d12f18276049822158f580d3d6de2965e7e05c12707003f56c23febb'];
    expect(entry.author).toBe('The scikit-learn developers');
    expect(entry.source_url).toBe('https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html');
    expect(entry.source_image_url).toBe('https://scikit-learn.org/stable/_images/sphx_glr_plot_polynomial_interpolation_002.png');
    expect(entry.source_asset).toBe(originals + 'spline-basis.png');
    expect(entry.source_page_snapshot).toBe(originals + 'spline-page.html');
    expect(entry.source_code).toBe(originals + 'spline-code.py');
    expect(entry.source_asset_sha256).toBe('a5c8285db6267f4a931501c52f16f95dfb8086e5a6f938c7b5ef3889ff012a25');
    expect(entry.sha256).toBe(hashes[index]);
    expect(entry.source_page_sha256).toBe('c67b7ed0d15ca0992f69f7e453945f501a78d0f79f2b780fef7e156cabf5e4f7');
    expect(entry.source_code_sha256).toBe('5cb63d64e206bc78e7eaa50caa3e989c419816e94fbcc63bb365d3e5d8ca4c5b');
    expect(entry.license).toBe('BSD-3-Clause');
    expect(entry.license_url).toBe('https://github.com/scikit-learn/scikit-learn/blob/1.9.1/COPYING');
    expect(entry.license_evidence).toBe(originals + 'sklearn-COPYING');
    expect(entry.license_evidence_sha256).toBe('50d6a9d340f19ab355609917993114daf5f47e3161067bcf34955bbd05cd9cb0');
    for (const [file, hash] of [
      [entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.source_code, entry.source_code_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]
    ]) expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([[1600,500],[665,500],[660,500]][index]);
    expect(readFileSync(resolve(repositoryRoot, entry.source_code!), 'utf8')).toContain('SPDX-License-Identifier: BSD-3-Clause');
    expect(readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8')).toContain('sphx_glr_plot_polynomial_interpolation_002.png');
    expect(readFileSync(resolve(repositoryRoot, 'site/public/licenses/scikit-learn-1.9.1-BSD-3-Clause.txt')).equals(readFileSync(resolve(repositoryRoot, entry.license_evidence!)))).toBe(true);
    expect(entry.derivation).toBe(index === 0 ? 'source-raster-copy' : 'source-raster-crop');
    expect(entry.metadata_status).toBe('verified');
    if (index === 0) expect(entry.crop_box_px).toBeUndefined();
    else expect(entry.crop_box_px).toEqual(index === 1 ? [135,0,800,500] : [805,0,1465,500]);
    requireNonEmptyString(entry.modifications, 'modifications');
    return;
  }
  const splitPrefix = '00 Учебник/Assets/Figures/curated/foundations-source-first/';
  const splitOriginals = '05 Источники/Originals/train-validation-test-2026-10-21/';
  const splitCases = [
    ['mooc-cross-validation-train-test.png', '195d5fd7ca626ccf95888bb72383a12a9b01e91526c232dfbfd4ccad5996b00a',
      'Assets/Sources/Scikit-learn MOOC/figures/cross_validation_train_test_diagram.png', 1200, 500],
    ['sklearn-group-kfold.png', '8a4680c6a641d4d7f29025a131dd344b11e017ac09553028a3819e6fbbb9f6cb',
      splitOriginals + 'group-kfold.png', 600, 300],
    ['sklearn-time-series-split.png', '3e57bcd142a9cd0a6e32ca9930358dd56a42952a93bf3c8b1d46a979a4c8df08',
      splitOriginals + 'time-series-split.png', 600, 300],
  ] as const;
  const splitCase = splitCases.find(([name]) => entry.asset === splitPrefix + name);
  if (splitCase) {
    const [name, hash, source, width, height] = splitCase;
    expect(entry.sha256).toBe(hash);
    expect(entry.source_asset).toBe(source);
    expect(entry.source_asset_sha256).toBe(hash);
    expect(sha256(resolve(repositoryRoot, source))).toBe(hash);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(hash);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([width, height]);
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    requireNonEmptyString(entry.modifications, 'unchanged copy description');
    if (name.startsWith('mooc-')) {
      expect(entry.author).toBe('INRIA, scikit-learn MOOC contributors');
      const blob = 'https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/';
      expect(entry.source_url).toBe(blob + 'figures/cross_validation_train_test_diagram.png');
      expect(entry.source_image_url).toBe(entry.source_url?.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/'));
      expect(entry.license).toBe('CC BY 4.0');
      expect(entry.license_url).toBe(blob + 'LICENSE');
      expect(entry.license_evidence).toBe(splitOriginals + 'mooc-LICENSE.txt');
      expect(entry.license_evidence_sha256).toBe('f5b745ef98087f531e719ee8ca6a96809444573ecc7173c6fa68eaad39b3cc3f');
    } else {
      expect(entry.author).toBe('The scikit-learn developers');
      expect(entry.source_url).toBe('https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html');
      expect(entry.source_image_url).toBe('https://scikit-learn.org/stable/_images/sphx_glr_plot_cv_indices_' + (name.includes('group') ? '007' : '013') + '.png');
      expect(entry.license).toBe('BSD-3-Clause');
      expect(entry.license_url).toBe('https://github.com/scikit-learn/scikit-learn/blob/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/COPYING');
      expect(entry.license_evidence).toBe(splitOriginals + 'sklearn-COPYING');
      expect(entry.license_evidence_sha256).toBe('50d6a9d340f19ab355609917993114daf5f47e3161067bcf34955bbd05cd9cb0');
      expect(entry.source_page_snapshot).toBe(splitOriginals + 'cv-indices.html');
      expect(entry.source_page_sha256).toBe('3319d4b5b4fa46fbba76773658b0b18d0befec06168536f3b28790ad1644a83f');
      expect(entry.source_code).toBe(splitOriginals + 'plot_cv_indices.py');
      expect(entry.source_code_sha256).toBe('56315d014009c15000559e8161b9c88c43c53022991efa8dbf74bad54b4d9b28');
      expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
      expect(sha256(resolve(repositoryRoot, entry.source_code!))).toBe(entry.source_code_sha256);
      expect(readFileSync(resolve(repositoryRoot, entry.source_code!), 'utf8')).toContain('SPDX-License-Identifier: BSD-3-Clause');
      expect(readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8')).toContain(entry.source_image_url!.split('/').at(-1));
    }
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/agent-security-2026-10-21/')) {
    const prefix = '00 Учебник/Assets/Figures/curated/agent-security-2026-10-21/';
    const expected = (JSON.parse(readFileSync(resolve(repositoryRoot, prefix + 'assets.json'), 'utf8')) as CuratedAsset[])
      .find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'license', 'license_url', 'source_url', 'source_image_url',
      'source_asset', 'source_asset_sha256', 'source_page', 'render_page_size_px',
      'source_page_snapshot', 'source_page_sha256', 'sha256', 'license_evidence',
      'license_evidence_sha256', 'derivation', 'metadata_status', 'provenance_confirmation'] as const)
      expect(entry[key]).toEqual(expected![key]);
    expect(entry.crop_box_px).toBeUndefined();
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    if (entry.asset.endsWith('m12-p13.png')) {
      expect(entry.source_page).toBe(13);
      expect(entry.derivation).toBe('pdf-page-render');
      expect(entry.rights_status).toBe('permission-recorded');
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([2400, 1350]);
    } else if (entry.asset.endsWith('agentpoison-overview.png')) {
      expect(entry.source_image_url).toBe('https://arxiv.org/html/2407.12784v1/overview_v5.png');
      expect(entry.license).not.toContain('CC BY');
      expect(entry.rights_status).toBe('permission-recorded');
      expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
      expect(readFileSync(resolve(repositoryRoot, entry.asset))
        .equals(readFileSync(resolve(repositoryRoot, entry.source_asset!)))).toBe(true);
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([2038, 1145]);
    } else {
      expect(entry.license).toBe('CC BY 4.0');
      expect(readFileSync(resolve(repositoryRoot, entry.asset))
        .equals(readFileSync(resolve(repositoryRoot, entry.source_asset!)))).toBe(true);
      for (const [file, hash] of [[entry.license_evidence, entry.license_evidence_sha256],
        [entry.source_page_snapshot, entry.source_page_sha256]])
        expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
      expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
        .toContain('creativecommons.org/licenses/by/4.0/');
      expect(readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8'))
        .toContain(entry.source_image_url!.split('/').at(-1));
    }
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/scientific-discovery-2026-10-21/')) {
    const archive = '05 Источники/Originals/scientific-discovery-2026-10-21/';
    const name = entry.asset.split('/').at(-1)!;
    const originals: Record<string, [string, string]> = {
      'lasr-overview.svg': ['2409.09359v3', 'neurips24-lasr-figure-1-overview-loopy-2.svg'],
      'lasr-evolution.svg': ['2409.09359v3', 'neurips24-lasr-figure-2-architecture-1.svg'],
      'lasr-ablations.svg': ['2409.09359v3', 'iter_graph.svg'],
      'escher-overview.svg': ['2504.00185v1', 'CVPR25-concept-evolution-figure-1-overview-5.svg'],
      'escher-bird-concepts.svg': ['2504.00185v1', 'cvpr25-concept-evol-qualitative-figure-4.svg'],
      'escher-history-prompt.svg': ['2504.00185v1', 'cvpr25-escher-history-prompt.svg'],
    };
    expect(originals[name]).toBeDefined();
    const [paper, upstream] = originals[name]!;
    const source = `https://arxiv.org/html/${paper}`;
    expect(entry.source_url).toMatch(new RegExp('^' + source.replaceAll('.', '\\.') + '#S[0-9]+\\.F[0-9]+$'));
    expect(entry.source_image_url).toBe(`${source}/${upstream}`);
    expect(entry.source_asset).toBe(archive + name);
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(readFileSync(resolve(repositoryRoot, entry.asset))
      .equals(readFileSync(resolve(repositoryRoot, entry.source_asset!)))).toBe(true);
    expect(entry.sha256).toBe(entry.source_asset_sha256);
    for (const [file, hash] of [[entry.license_evidence, entry.license_evidence_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain('creativecommons.org/licenses/by/4.0/');
    expect(readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8'))
      .toContain(`${paper}/${upstream}`);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/formal-math-2026-10-21/')) {
    const archive = '05 Источники/Originals/formal-math-2026-10-21/';
    const expected = (JSON.parse(readFileSync(resolve(repositoryRoot,
      entry.asset.slice(0, entry.asset.lastIndexOf('/')) + '/assets.json'), 'utf8')) as CuratedAsset[])
      .find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'license', 'license_url', 'source_url', 'source_image_url',
      'source_asset', 'source_asset_sha256', 'source_page', 'source_page_snapshot',
      'source_page_sha256', 'crop_box_px', 'render_page_size_px', 'sha256',
      'license_evidence', 'license_evidence_sha256', 'derivation'] as const)
      expect(entry[key]).toEqual(expected![key]);
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    if (entry.source_asset) {
      expect(entry.source_asset).toBe(archive + 'leaneuclid-pmlr.pdf');
      const overview = entry.asset.endsWith('autoformalization.png');
      expect(entry.source_page).toBe(overview ? 2 : 5);
      expect(entry.crop_box_px).toEqual(overview ? [260, 320, 2760, 1950] : [1535, 600, 2760, 2425]);
      expect(entry.render_page_size_px).toEqual([3091, 4000]);
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual(overview ? [2500, 1630] : [1225, 1825]);
      for (const [file, hash] of [[entry.source_asset, entry.source_asset_sha256],
        [entry.source_page_snapshot!, entry.source_page_sha256]])
        expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    } else {
      expect(entry.source_url).toBe('https://arxiv.org/abs/2407.10040v1');
      expect(entry.sha256).toBe(entry.source_asset_sha256);
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual(
        entry.asset.endsWith('pipeline.png') ? [8095, 3007] : [5785, 3044]);
      expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
        .toContain('creativecommons.org/licenses/by/4.0/');
    }
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/coding-web-computer-use-2026-10/')) {
    const archive = '05 Источники/Originals/coding-web-computer-use-2026-10-21/';
    const refs = JSON.parse(readFileSync(resolve(repositoryRoot, archive + 'references.json'), 'utf8')) as { figures: CuratedAsset[] };
    const expected = refs.figures.find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'license', 'license_url', 'source_url', 'source_asset',
      'source_asset_sha256', 'source_page', 'source_page_snapshot', 'source_page_sha256',
      'crop_box_px', 'render_page_size_px', 'sha256', 'license_evidence', 'license_evidence_sha256'] as const)
      expect(entry[key]).toEqual(expected![key]);
    expect(entry.asset.endsWith('/taco-cota-examples.png')).toBe(true);
    expect(entry.source_asset).toBe(archive + 'taco.pdf');
    expect(entry.source_page).toBe(2);
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.modifications).toContain('Complete original numbered figure and English caption');
    for (const [file, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain('href="http://creativecommons.org/licenses/by/4.0/"');
    expect(pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!))).toEqual([3091, 4000]);
    expect(entry.crop_box_px).toEqual([285, 365, 2805, 2600]);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([2520, 2235]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/agent-evaluation-2026-10/')) {
    const archive = '05 Источники/Originals/agent-evaluation-2026-10-20/';
    const refs = JSON.parse(readFileSync(resolve(repositoryRoot, archive + 'references.json'), 'utf8')) as { figures: CuratedAsset[] };
    const expected = refs.figures.find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'license', 'license_url', 'source_url', 'source_asset',
      'source_asset_sha256', 'source_page', 'source_page_snapshot', 'source_page_sha256',
      'crop_box_px', 'render_page_size_px', 'sha256', 'license_evidence', 'license_evidence_sha256'] as const)
      expect(entry[key]).toEqual(expected![key]);
    const spec = entry.asset.endsWith('/swe-bench-construction.png') ? ['swe-bench', 2]
      : entry.asset.endsWith('/swe-bench-evaluation.png') ? ['swe-bench', 20]
      : entry.asset.endsWith('/tau-bench-task-components.png') ? ['tau-bench', 4]
      : entry.asset.endsWith('/paperbench-rubric-tree.png') ? ['paperbench', 3]
      : entry.asset.endsWith('/agentbench-environments-results.png') ? ['agentbench', 1] : null;
    expect(spec).not.toBeNull();
    expect(entry.source_asset).toBe(archive + spec![0] + '.pdf');
    expect(entry.source_page).toBe(spec![1]);
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.modifications).toContain('Complete original numbered figure and English caption');
    for (const [file, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain('href="http://creativecommons.org/licenses/by/4.0/"');
    expect(pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!))).toEqual(entry.render_page_size_px);
    const [x1, y1, x2, y2] = entry.crop_box_px as number[];
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([x2!-x1!, y2!-y1!]);
    expect(x2!-x1!).toBeGreaterThanOrEqual(1200);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/agent-memory-planning-2026-10/')) {
    const archive = '05 Источники/Originals/agent-memory-planning-2026-10-20/';
    const refs = JSON.parse(readFileSync(resolve(repositoryRoot, archive + 'references.json'), 'utf8')) as { figures: CuratedAsset[] };
    const expected = refs.figures.find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'license', 'license_url', 'source_url', 'source_asset',
      'source_asset_sha256', 'source_page', 'source_page_snapshot', 'source_page_sha256',
      'crop_box_px', 'render_page_size_px', 'sha256', 'license_evidence', 'license_evidence_sha256'] as const)
      expect(entry[key]).toEqual(expected![key]);
    const tot = entry.asset.endsWith('/tot-game24-search.png');
    expect(entry.source_asset).toBe(archive + (tot ? 'tree-of-thoughts.pdf' : 'reflexion.pdf'));
    expect(entry.source_page).toBe(tot ? 5 : 4);
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.modifications).toContain('Complete original numbered figure and English caption');
    for (const [file, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain('License: CC BY 4.0');
    expect(pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!))).toEqual(entry.render_page_size_px);
    const [x1, y1, x2, y2] = entry.crop_box_px as number[];
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([x2!-x1!, y2!-y1!]);
    expect(x2!-x1!).toBeGreaterThanOrEqual(2000);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/agent-harness-2026-10/')) {
    const archive = '05 Источники/Originals/agent-harness-2026-10-20/';
    const refs = JSON.parse(readFileSync(resolve(repositoryRoot, archive + 'references.json'), 'utf8')) as { figures: CuratedAsset[] };
    const expected = refs.figures.find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'license', 'license_url', 'source_url', 'source_asset',
      'source_asset_sha256', 'source_page', 'source_page_snapshot', 'source_page_sha256',
      'crop_box_px', 'render_page_size_px', 'sha256', 'license_evidence', 'license_evidence_sha256'] as const)
      expect(entry[key]).toEqual(expected![key]);
    const swe = entry.asset.endsWith('/swe-agent-prompt-workflow.png');
    expect(entry.source_asset).toBe(swe ? archive + 'swe-agent.pdf'
      : '05 Источники/Papers/Source PDFs/lost-in-the-middle-tacl-2024.pdf');
    expect(entry.source_url).toBe(swe ? 'https://arxiv.org/abs/2405.15793v3'
      : 'https://aclanthology.org/2024.tacl-1.9/');
    expect(entry.source_page).toBe(swe ? 39 : entry.asset.endsWith('/lost-middle-position-effect.png') ? 2 : 4);
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.modifications).toContain('Complete original numbered figure(s) and English caption(s)');
    for (const [file, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain(swe ? 'creativecommons.org/licenses/by/4.0/' : 'Distributed under a CC-BY 4.0 license.');
    expect(pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!))).toEqual(entry.render_page_size_px);
    const [x1, y1, x2, y2] = entry.crop_box_px as number[];
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([x2!-x1!, y2!-y1!]);
    expect(x2!-x1!).toBeGreaterThanOrEqual(1000);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/tool-use-2026-10/')) {
    const archive = '05 Источники/Originals/tool-use-2026-10-20/';
    const refs = JSON.parse(readFileSync(resolve(repositoryRoot, archive + 'references.json'), 'utf8')) as { figures: CuratedAsset[] };
    const expected = refs.figures.find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'license', 'license_url', 'source_url', 'source_asset',
      'source_asset_sha256', 'source_page', 'source_page_snapshot', 'source_page_sha256',
      'crop_box_px', 'render_page_size_px', 'sha256', 'license_evidence', 'license_evidence_sha256'] as const)
      expect(entry[key]).toEqual(expected![key]);
    expect(entry.source_asset).toBe(archive + 'react.pdf');
    expect(entry.source_url).toBe('https://arxiv.org/abs/2210.03629v3');
    expect(entry.source_page).toBe(2);
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.modifications).toContain('Complete numbered panel from original Figure 1');
    for (const [file, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain('creativecommons.org/licenses/by/4.0/');
    expect(pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!))).toEqual(entry.render_page_size_px);
    const [x1, y1, x2, y2] = entry.crop_box_px as number[];
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([x2!-x1!, y2!-y1!]);
    expect(x2!-x1!).toBeGreaterThanOrEqual(2000);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/chameleon-2026-10/')) {
    const archive = '05 Источники/Originals/chameleon-2026-10-20/';
    const refs = JSON.parse(readFileSync(resolve(repositoryRoot, archive + 'references.json'), 'utf8')) as { figures: CuratedAsset[] };
    const expected = refs.figures.find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'license', 'license_url', 'source_url', 'source_asset',
      'source_asset_sha256', 'source_page', 'source_page_snapshot', 'source_page_sha256',
      'crop_box_px', 'render_page_size_px', 'sha256', 'license_evidence', 'license_evidence_sha256'] as const)
      expect(entry[key]).toEqual(expected![key]);
    expect(entry.source_asset).toBe(archive + 'chameleon.pdf');
    expect(entry.source_url).toBe('https://arxiv.org/abs/2405.09818v2');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.modifications).toContain('Complete original figure and English caption');
    for (const [file, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!))).toEqual(entry.render_page_size_px);
    const [x1, y1, x2, y2] = entry.crop_box_px as number[];
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([x2!-x1!, y2!-y1!]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/diffusion-flow-2026-10/')) {
    const archive = '05 Источники/Originals/diffusion-flow-2026-10-20/';
    const refs = JSON.parse(readFileSync(resolve(repositoryRoot, archive + 'references.json'), 'utf8')) as {
      figures: CuratedAsset[];
    };
    const expected = refs.figures.find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'license', 'license_url', 'source_url', 'source_asset',
      'source_asset_sha256', 'source_page', 'source_page_snapshot', 'source_page_sha256',
      'crop_box_px', 'render_page_size_px', 'sha256', 'license_evidence', 'license_evidence_sha256'] as const)
      expect(entry[key]).toEqual(expected![key]);
    const mit = entry.asset.split('/').pop()!.startsWith('mit-');
    const dit = entry.asset.endsWith('/dit-architecture.png');
    expect(entry.source_asset).toBe(archive + (mit ? 'mit-notes-v3.pdf' : dit ? 'dit.pdf' : 'sd3-pmlr.pdf'));
    expect(entry.source_url).toBe(mit ? 'https://arxiv.org/abs/2506.02070v3'
      : dit ? 'https://arxiv.org/abs/2212.09748v2' : 'https://proceedings.mlr.press/v235/esser24a.html');
    expect(entry.license).toBe(mit ? 'CC BY-NC-ND 4.0' : 'CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/' + (mit ? 'by-nc-nd' : 'by') + '/4.0/');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    requireNonEmptyString(entry.modifications, 'diffusion figure reproduction description');
    if (mit) expect(entry.modifications).toContain('Unmodified complete figure and original English caption');
    for (const [file, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!))).toEqual(entry.render_page_size_px);
    const [x1, y1, x2, y2] = entry.crop_box_px as number[];
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([x2!-x1!, y2!-y1!]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/vlm-evaluation-review-2026-10/')) {
    const archive = '05 Источники/Originals/vlm-evaluation-serving-2026-10-20/';
    const refs = JSON.parse(readFileSync(resolve(repositoryRoot, archive + 'references.json'), 'utf8')) as {
      figures: CuratedAsset[]; downloaded: Array<{ path: string; sha256: string }>;
    };
    const expected = refs.figures.find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'license', 'license_url', 'source_url', 'source_asset',
      'source_asset_sha256', 'source_page', 'source_page_snapshot', 'source_page_sha256',
      'crop_box_px', 'render_page_size_px', 'sha256', 'license_evidence', 'license_evidence_sha256'] as const)
      expect(entry[key]).toEqual(expected![key]);
    const mmvet = entry.asset.endsWith('/mmvet-integrated-capabilities.png');
    const pope = entry.asset.endsWith('/pope-question-construction.png');
    const chair = entry.asset.endsWith('/chair-caption-errors.png');
    const donor = mmvet ? 'mmvet-pmlr' : pope ? 'pope' : chair ? 'chair' : 'epd-pmlr';
    expect(entry.source_asset).toBe(archive + donor + '.pdf');
    expect(entry.source_url).toBe(mmvet ? 'https://proceedings.mlr.press/v235/yu24o.html'
      : pope ? 'https://aclanthology.org/2023.emnlp-main.20/'
      : chair ? 'https://aclanthology.org/D18-1437/' : 'https://proceedings.mlr.press/v267/singh25d.html');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.license_evidence).toBe(archive + ((pope || chair)
      ? 'acl-copyright.html' : 'pmlr-license-agreement.pdf'));
    const evidence = refs.downloaded.find(row => row.path === entry.license_evidence);
    expect(evidence).toBeDefined();
    expect(entry.license_evidence_sha256).toBe(evidence!.sha256);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    requireNonEmptyString(entry.modifications, 'VLM evaluation figure modifications');
    for (const [file, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    if (pope || chair)
      expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
        .toContain('creativecommons.org/licenses/by/4.0/');
    else
      expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!)).subarray(0, 5).toString()).toBe('%PDF-');
    expect(readFileSync(resolve(repositoryRoot, entry.source_asset!)).subarray(0, 5).toString()).toBe('%PDF-');
    expect(pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!))).toEqual(entry.render_page_size_px);
    const [x1, y1, x2, y2] = entry.crop_box_px as number[];
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([x2!-x1!, y2!-y1!]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/omni-review-2026-10/')) {
    const archive = '05 Источники/Originals/omni-2026-10-20/';
    const refs = JSON.parse(readFileSync(resolve(repositoryRoot, archive + 'references.json'), 'utf8')) as {
      figures: CuratedAsset[]; downloaded: Array<{ path: string; sha256: string }>;
    };
    const expected = refs.figures.find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'license', 'license_url', 'source_url', 'source_asset',
      'source_asset_sha256', 'source_page', 'source_page_snapshot', 'source_page_sha256',
      'crop_box_px', 'render_page_size_px', 'sha256', 'license_evidence', 'license_evidence_sha256'] as const)
      expect(entry[key]).toEqual(expected![key]);
    const moshi = entry.asset.includes('/mimi-') || entry.asset.includes('/moshi-');
    const gens = entry.asset.endsWith('/gens-frame-selection.png');
    const qwen3 = entry.asset.endsWith('/qwen3omni-architecture.png');
    const donor = moshi ? 'moshi' : gens ? 'gens-acl' : qwen3 ? 'qwen3omni' : 'qwen25omni';
    expect(entry.source_asset).toBe(archive + donor + '.pdf');
    expect(entry.source_url).toBe(moshi ? 'https://arxiv.org/pdf/2410.00037v2'
      : gens ? 'https://aclanthology.org/2025.findings-acl.921.pdf'
      : qwen3 ? 'https://arxiv.org/pdf/2509.17765v1' : 'https://arxiv.org/pdf/2503.20215v1');
    const license = moshi ? 'by-nc-sa' : 'by';
    expect(entry.license).toBe(moshi ? 'CC BY-NC-SA 4.0' : 'CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/' + license + '/4.0/');
    expect(entry.license_evidence).toBe(archive + (gens ? 'acl-copyright.html' : donor + '-abstract.html'));
    const evidence = refs.downloaded.find(row => row.path === entry.license_evidence);
    expect(evidence).toBeDefined();
    expect(entry.license_evidence_sha256).toBe(evidence!.sha256);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    requireNonEmptyString(entry.modifications, 'omni figure modifications');
    for (const [file, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain('creativecommons.org/licenses/' + license + '/4.0/');
    expect(readFileSync(resolve(repositoryRoot, entry.source_asset!)).subarray(0, 5).toString()).toBe('%PDF-');
    expect(pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!))).toEqual(entry.render_page_size_px);
    const [x1, y1, x2, y2] = entry.crop_box_px as number[];
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([x2!-x1!, y2!-y1!]);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/documents-review-2026-10/')) {
    const archive = '05 Источники/Originals/documents-2026-10-20/';
    const refs = JSON.parse(readFileSync(resolve(repositoryRoot, archive + 'references.json'), 'utf8')) as {
      figures: CuratedAsset[];
      downloaded: Array<{ path: string; sha256: string }>;
    };
    expect(entry.metadata_status).toBe('verified');
    requireNonEmptyString(entry.modifications, 'document figure modifications');
    for (const [path, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    if (entry.asset.endsWith('/donut-overview.png')) {
      const pin = '4cfcf972560e1a0f26eb3e294c8fc88a0d336626';
      expect(entry.derivation).toBe('source-raster-copy');
      expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
      expect(entry.source_asset).toBe(archive + 'donut-overview.png');
      expect(entry.sha256).toBe(entry.source_asset_sha256);
      expect(entry.commit).toBe(pin);
      expect(entry.source_url).toBe('https://github.com/clovaai/donut/blob/' + pin + '/misc/overview.png');
      expect(entry.license).toBe('MIT');
      expect(entry.license_url).toBe('https://github.com/clovaai/donut/blob/' + pin + '/LICENSE');
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([4237, 1007]);
      expect(sha256(resolve(repositoryRoot, 'site/public/licenses/donut-4cfcf972-MIT.txt')))
        .toBe(sha256(resolve(repositoryRoot, archive + 'donut-LICENSE.txt')));
      expect(sha256(resolve(repositoryRoot, 'site/public/licenses/donut-4cfcf972-NOTICE.txt')))
        .toBe(sha256(resolve(repositoryRoot, archive + 'donut-NOTICE.txt')));
      return;
    }
    const expected = refs.figures.find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'license', 'source_asset', 'source_asset_sha256',
      'source_page', 'source_page_snapshot', 'source_page_sha256', 'crop_box_px',
      'render_page_size_px', 'sha256'] as const)
      expect(entry[key]).toEqual(expected![key]);
    const layout = entry.asset.endsWith('/layoutlmv3-objectives.png');
    expect(entry.source_asset).toBe(archive + (layout ? 'layoutlm.pdf' : 'pix2struct.pdf'));
    expect(entry.source_url).toBe(layout
      ? 'https://arxiv.org/pdf/2204.08387v3'
      : 'https://proceedings.mlr.press/v202/lee23g/lee23g.pdf');
    expect(entry.license).toBe(layout ? 'CC BY-NC-SA 4.0' : 'CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/' + (layout ? 'by-nc-sa' : 'by') + '/4.0/');
    expect(entry.license_evidence).toBe(archive + (layout ? 'layoutlm-abstract.html' : 'pix2struct-abstract.html'));
    const evidence = refs.downloaded.find(row => row.path === entry.license_evidence);
    expect(evidence).toBeDefined();
    expect(entry.license_evidence_sha256).toBe(evidence!.sha256);
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(evidence!.sha256);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain('creativecommons.org/licenses/' + (layout ? 'by-nc-sa' : 'by') + '/4.0/');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    expect(pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!))).toEqual(entry.render_page_size_px);
    const [x1, y1, x2, y2] = entry.crop_box_px as number[];
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([x2!-x1!, y2!-y1!]);
    return;
  }
  const isConnectors = entry.asset.startsWith('00 Учебник/Assets/Figures/curated/connectors-review-2026-10/');
  const isResolution = entry.asset.startsWith('00 Учебник/Assets/Figures/curated/resolution-review-2026-10/');
  if (isConnectors || isResolution || entry.asset.startsWith('00 Учебник/Assets/Figures/curated/vision-encoders-review-2026-10/')) {
    const family = isResolution ? 'resolution-review' : isConnectors ? 'connectors-review' : 'vision-encoders-review';
    const manifest = JSON.parse(readFileSync(resolve(repositoryRoot,
      `00 Учебник/Assets/Figures/curated/${family}-2026-10/assets.json`), 'utf8')) as CuratedAsset[];
    const expected = manifest.find(row => row.asset === entry.asset);
    expect(expected).toBeDefined();
    for (const key of ['author', 'source_url', 'source_asset', 'source_asset_sha256',
      'source_page', 'source_page_snapshot', 'source_page_sha256', 'crop_box_px',
      'render_page_size_px', 'license_evidence', 'license_evidence_sha256', 'sha256'] as const)
      expect(entry[key]).toEqual(expected![key]);
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.source_asset).toMatch(isResolution
      ? /^05 Источники\/Originals\/resolution-2026-10-20\/pixtral\.pdf$/
      : isConnectors
        ? /^05 Источники\/Originals\/connectors-2026-10-20\/blip2\.pdf$/
        : /^05 Источники\/Originals\/vision-encoders-2026-10-20\/(?:clip|siglip)\.pdf$/);
    for (const [file, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.source_asset!)).subarray(0, 5).toString()).toBe('%PDF-');
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toMatch(/creativecommons\.org\/licenses\/by\/4\.0/);
    expect(pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!))).toEqual(entry.render_page_size_px);
    const box = entry.crop_box_px as number[];
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([box[2] - box[0], box[3] - box[1]]);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (/\/source-first-60-64\/hf-rag-(?:evaluation-workflow|settings-accuracy)\.png$/.test(entry.asset)) {
    const prefix = '05 Источники/Originals/rag-2026-10-20/';
    const isFlow = entry.asset.endsWith('evaluation-workflow.png');
    expect(entry.source_asset).toBe(prefix + (isFlow ? 'hf-flow.png' : 'hf-accuracy.png'));
    expect(entry.source_image_url).toBe('https://huggingface.co/datasets/huggingface/cookbook-images/resolve/0b825cadd3c8467717f393bcbe38b518dec75e8e/' +
      (isFlow ? 'RAG_workflow.png' : 'RAG_settings_accuracy.png'));
    expect(entry.source_url).toBe('https://huggingface.co/learn/cookbook/en/rag_evaluation');
    expect(entry.commit).toBe('0b825cadd3c8467717f393bcbe38b518dec75e8e');
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.sha256);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.license).toBe('open educational material; reuse confirmed by repository owner');
    expect(entry.license_evidence).toBe(prefix + 'legacy-figure-record.json');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.provenance_confirmation).toBe('user-confirmed-open-materials');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.metadata_status).toBe('verified');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/rag-review-2026-10/')) {
    const prefix = '05 Источники/Originals/rag-2026-10-20/';
    const manifest = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/rag-review-2026-10/assets.json'), 'utf8')) as CuratedAsset[];
    const original = manifest.find(row => row.asset === entry.asset);
    expect(original).toBeDefined();
    for (const key of ['author', 'source_url', 'source_asset', 'source_asset_sha256',
      'source_page', 'source_page_snapshot', 'source_page_sha256', 'crop_box_px',
      'render_page_size_px', 'license', 'license_url', 'license_evidence',
      'license_evidence_sha256', 'derivation', 'provenance_confirmation', 'sha256'] as const)
      expect(entry[key]).toEqual(original![key]);
    expect(entry.metadata_status).toBe('verified');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.source_asset).toMatch(/^05 Источники\/Originals\/rag-2026-10-20\/[\w-]+\.pdf$/);
    for (const [file, hash] of [
      [entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]
    ]) expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.source_asset!)).subarray(0, 5).toString()).toBe('%PDF-');
    const size = pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!));
    expect(size).toEqual(entry.render_page_size_px);
    const box = entry.crop_box_px as number[];
    expect(box).toHaveLength(4);
    expect(box.every(Number.isInteger)).toBe(true);
    expect(box[0]).toBeGreaterThanOrEqual(0);
    expect(box[1]).toBeGreaterThanOrEqual(0);
    expect(box[2]).toBeLessThanOrEqual(size[0]!);
    expect(box[3]).toBeLessThanOrEqual(size[1]!);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([box[2] - box[0], box[3] - box[1]]);
    if (entry.asset.endsWith('/rag-sequence-token.png')) {
      expect(entry.license).toBe('open educational material; reuse confirmed by repository owner');
      expect(entry.license_evidence).toBe(prefix + 'legacy-figure-record.json');
      expect(JSON.parse(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).scope).toContain('not independently verified');
    } else {
      expect(entry.license).toBe('CC BY 4.0');
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
      const evidence = readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8');
      expect(evidence).toMatch(/creativecommons\.org\/licenses\/by\/4\.0|Creative Commons 4\.0 BY/);
    }
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/source-first-60-64/sbert-bi-vs-cross-encoder.png') {
    const commit = '8a880ec7f1524640619fe382940ee319144f74ca';
    expect(entry.commit).toBe(commit);
    expect(entry.source_url).toBe(`https://github.com/huggingface/sentence-transformers/blob/${commit}/examples/cross_encoder/applications/README.md`);
    expect(entry.source_image_url).toBe(`https://raw.githubusercontent.com/huggingface/sentence-transformers/${commit}/docs/img/Bi_vs_Cross-Encoder.png`);
    expect(entry.source_asset).toBe('05 Источники/Originals/reranking-2026-10-20/bi-cross-encoder.png');
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.license).toBe('Apache-2.0');
    expect(entry.license_url).toBe(`https://github.com/huggingface/sentence-transformers/blob/${commit}/LICENSE`);
    expect(entry.license_evidence).toBe('05 Источники/Originals/embeddings-2026-10-20/sbert-LICENSE');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.derivation).toBe('source-raster-copy');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/reranking-review-2026-10/')) {
    const prefix = '05 Источники/Originals/reranking-2026-10-20/';
    const name = entry.asset.split('/').at(-1)!;
    const expected: Record<string, [string, number, number[], string]> = {
      'colbert-paradigms.png': ['colbert-v1', 2, [215, 320, 2345, 940], 'https://arxiv.org/abs/2004.12832v2'],
      'colbert-maxsim.png': ['colbert-qa', 2, [255, 205, 1085, 560], 'https://aclanthology.org/2021.tacl-1.55/'],
      'colbert-supervision.png': ['colbert-qa', 2, [255, 560, 1085, 853], 'https://aclanthology.org/2021.tacl-1.55/'],
      'rankgpt-instructions.png': ['rankgpt', 3, [295, 290, 1250, 1045], 'https://aclanthology.org/2023.emnlp-main.923/'],
      'rankgpt-windows.png': ['rankgpt', 3, [1300, 295, 2290, 655], 'https://aclanthology.org/2023.emnlp-main.923/'],
      'calibration-reliability.png': ['calibration', 1, [1090, 602, 1955, 1505], 'https://proceedings.mlr.press/v70/guo17a.html']
    };
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    if (name === 'retrieve-rerank.png') {
      expect(entry.source_url).toBe('https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html');
      expect(entry.source_asset).toBe(prefix + 'retrieve-rerank.png');
      expect(entry.source_asset_sha256).toBe(entry.sha256);
      expect(entry.commit).toBe('8a880ec7f1524640619fe382940ee319144f74ca');
      expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/huggingface/sentence-transformers/8a880ec7f1524640619fe382940ee319144f74ca/docs/img/InformationRetrieval.png');
      expect(entry.source_code).toBe(prefix + 'st-retrieve-rerank.md');
      expect(sha256(resolve(repositoryRoot, entry.source_code!))).toBe(entry.source_code_sha256);
      expect(readFileSync(resolve(repositoryRoot, entry.source_code!), 'utf8')).toContain('docs/img/InformationRetrieval.png');
      expect(entry.license).toBe('Apache-2.0');
      expect(entry.license_evidence).toBe('05 Источники/Originals/embeddings-2026-10-20/sbert-LICENSE');
      expect(entry.derivation).toBe('source-raster-copy');
      expect(entry.crop_box_px).toBeUndefined();
      expect(entry.provenance_confirmation).toBe('source-license-verified');
    } else {
      expect(expected[name]).toBeDefined();
      const [pdf, page, box, url] = expected[name];
      expect(entry.source_url).toBe(url);
      expect(entry.source_asset).toBe(prefix + pdf + '.pdf');
      expect(readFileSync(resolve(repositoryRoot, entry.source_asset!)).subarray(0, 5).toString()).toBe('%PDF-');
      expect(entry.source_page).toBe(page);
      expect(entry.source_page_snapshot).toBe(prefix + pdf + '-page-0' + page + '.png');
      expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
      expect(entry.render_page_size_px).toEqual(pngDimensions(resolve(repositoryRoot, entry.source_page_snapshot!)));
      expect(entry.crop_box_px).toEqual(box);
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([box[2] - box[0], box[3] - box[1]]);
      expect(entry.derivation).toBe('pdf-page-render-crop');
      if (name === 'colbert-paradigms.png') {
        expect(entry.license).toBe('open educational material; reuse confirmed by repository owner');
        expect(entry.provenance_confirmation).toBe('user-confirmed-open-materials');
        expect(entry.license_evidence).toBe(prefix + 'legacy-figure-record.json');
        expect(JSON.parse(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).scope).toContain('not independently verified');
      } else {
        expect(entry.license).toBe('CC BY 4.0');
        expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
        expect(entry.provenance_confirmation).toBe('source-license-verified');
        expect(entry.license_evidence).toBe(pdf === 'colbert-qa' ? prefix + 'colbert-qa.pdf'
          : pdf === 'calibration' ? prefix + 'pmlr-license-agreement.html'
          : '05 Источники/Originals/retrieval-2026-10-20/acl-copyright.html');
      }
    }
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/retrieval-review-2026-10/')) {
    const prefix = '05 Источники/Originals/retrieval-2026-10-20/';
    const name = entry.asset.split('/').at(-1)!;
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    if (name === 'hyde-pipeline.png') {
      expect(entry.source_url).toBe('https://aclanthology.org/2023.acl-long.99/');
      expect(entry.source_asset).toBe(prefix + 'hyde-acl.pdf');
      expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
      expect(readFileSync(resolve(repositoryRoot, entry.source_asset!)).subarray(0, 5).toString()).toBe('%PDF-');
      expect(entry.source_page).toBe(2);
      expect(entry.source_page_snapshot).toBe(prefix + 'hyde-page2.png');
      expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
      expect(entry.render_page_size_px).toEqual([1819, 2573]);
      expect(entry.crop_box_px).toEqual([200, 200, 1630, 590]);
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([1430, 390]);
      expect(entry.license_evidence).toBe(prefix + 'acl-copyright.html');
      expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).toContain('Creative Commons 4.0 BY');
      expect(entry.license).toBe('CC BY 4.0');
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
      expect(entry.derivation).toBe('pdf-page-render-crop');
      expect(entry.provenance_confirmation).toBe('source-license-verified');
    } else {
      const cases: Record<string, [string, string]> = {
        'hnsw.png': ['hnsw', 'hnsw.png'],
        'ivf-build.png': ['ivf-flat', 'IVF-FLAT-workflow.png'],
        'ivf-search.png': ['ivf-flat', 'IVF-FLAT-workflow-2.png'],
        'pq-codebooks.png': ['ivf-pq', 'ivf-pq-1.png'],
        'pq-search.png': ['ivf-pq', 'ivf-pq-2.png']
      };
      expect(cases[name]).toBeDefined();
      const [doc, image] = cases[name];
      expect(entry.source_url).toBe(`https://milvus.io/docs/${doc}.md`);
      expect(entry.source_image_url).toBe(`https://milvus-docs.s3.us-west-2.amazonaws.com/assets/${image}`);
      expect(entry.commit).toBe('9acdc99959e188a54be1df512ea3806c3110ef59');
      expect(entry.source_asset).toBe(prefix + name);
      expect(entry.source_asset_sha256).toBe(entry.sha256);
      expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.sha256);
      expect(entry.source_code).toBe(prefix + 'milvus-' + doc + '.md');
      expect(sha256(resolve(repositoryRoot, entry.source_code!))).toBe(entry.source_code_sha256);
      expect(readFileSync(resolve(repositoryRoot, entry.source_code!), 'utf8')).toContain(entry.source_image_url);
      expect(entry.license).toBe('Apache-2.0');
      expect(entry.license_url).toBe('https://github.com/milvus-io/milvus-docs/blob/9acdc99959e188a54be1df512ea3806c3110ef59/LICENSE');
      expect(entry.license_evidence).toBe(prefix + 'milvus-LICENSE');
      expect(entry.derivation).toBe('source-raster-copy');
      expect(entry.provenance_confirmation).toBe('official-milvus-documentation-asset');
      expect(entry.crop_box_px).toBeUndefined();
    }
    validateRenderedPngBytes(readFileSync(resolve(repositoryRoot, entry.asset)));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/source-first-60-64/openqa-')) {
    const name = entry.asset.split('/').at(-1)!;
    const page = {'openqa-tfidf-retriever.png': 5, 'openqa-retriever-reader.png': 4}[name];
    expect(page).toBeDefined();
    expect(entry.source_page).toBe(page);
    expect(entry.source_url).toBe(`https://github.com/danqi/acl2020-openqa-tutorial/blob/3c82cf3bfa7f7f26aa4016cf4c48b83f1a4eaad0/slides/part4-retriever-reader.pdf#page=${page}`);
    expect(entry.source_asset).toBe('05 Источники/Originals/retrieval-2026-10-20/openqa-part4.pdf');
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(entry.derivation).toBe('pdf-page-render');
    expect(entry.provenance_confirmation).toBe('user-confirmed-open-materials');
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([2560, 1920]);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/embeddings-review-2026-10/')) {
    const prefix = '05 Источники/Originals/embeddings-2026-10-20/';
    const name = entry.asset.split('/').at(-1)!;
    const sbert = '8a880ec7f1524640619fe382940ee319144f74ca';
    const simcse = '49fa580a853752ede55b8c76d9debf748e214d3f';
    const hf = '1bade87e4c4e41d80d5f72ba2d7b38ab3194f044';
    const cases: Record<string, [string, string, string, string, string]> = {
      'bi-cross-encoder.png': [sbert, 'Apache-2.0', 'sbert-LICENSE', 'cross-encoder.html', `https://raw.githubusercontent.com/huggingface/sentence-transformers/${sbert}/docs/img/Bi_vs_Cross-Encoder.png`],
      'multiple-negatives.png': [sbert, 'Apache-2.0', 'sbert-LICENSE', 'msmarco.md', `https://raw.githubusercontent.com/huggingface/sentence-transformers/${sbert}/docs/img/MultipleNegativeRankingLoss.png`],
      'distillation.png': [sbert, 'Apache-2.0', 'sbert-LICENSE', 'cross-distillation.md', `https://raw.githubusercontent.com/huggingface/sentence-transformers/${sbert}/docs/img/msmarco-training-ce-distillation.png`],
      'simcse.png': [simcse, 'MIT', 'simcse-LICENSE', 'simcse-README.md', `https://raw.githubusercontent.com/princeton-nlp/SimCSE/${simcse}/figure/model.png`],
      'matryoshka-results.png': [hf, 'CC BY-NC-SA 4.0', 'hf-images-README.md', 'matryoshka.md', `https://huggingface.co/datasets/huggingface/documentation-images/resolve/${hf}/blog/matryoshka/results.png`]
    };
    expect(cases[name]).toBeDefined();
    const [pin, license, evidence, page, url] = cases[name];
    const manifest = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/embeddings-review-2026-10/assets.json'), 'utf8')) as CuratedAsset[];
    const original = manifest.find(row => row.asset === entry.asset)!;
    expect(original).toBeDefined();
    for (const key of ['source_url', 'source_image_url', 'license_url'] as const)
      expect(entry[key]).toBe(original[key]);
    expect(entry.source_image_url).toBe(url);
    expect(entry.commit).toBe(pin);
    expect(entry.license).toBe(license);
    expect(entry.source_asset).toBe(prefix + name);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.sha256);
    expect(entry.license_evidence).toBe(prefix + evidence);
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.source_page_snapshot).toBe(prefix + page);
    expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.crop_box_px).toBeUndefined();
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/source-first-60-64/dpr-')) {
    const names = ['dpr-dual-encoder-and-loss.png', 'dpr-positive-and-negative-examples.png', 'dpr-in-batch-negatives.png'];
    const page = names.indexOf(entry.asset.split('/').at(-1)!) + 20;
    expect(page).toBeGreaterThanOrEqual(20);
    expect(entry.source_page).toBe(page);
    expect(entry.source_url).toBe(`https://github.com/danqi/acl2020-openqa-tutorial/blob/3c82cf3bfa7f7f26aa4016cf4c48b83f1a4eaad0/slides/part5-dense-retriever-e2e-training.pdf#page=${page}`);
    expect(entry.source_asset).toBe('05 Источники/Originals/embeddings-2026-10-20/openqa-part5.pdf');
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(entry.derivation).toBe('pdf-page-render');
    expect(entry.provenance_confirmation).toBe('user-confirmed-open-materials');
    expect(entry.license).toBe('open educational material; reuse confirmed by repository owner');
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([2560, 1920]);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/responsible-review-2026-10/')) {
    const prefix = '05 Источники/Originals/responsible-2026-10-20/';
    const name = entry.asset.split('/').at(-1)!;
    const cases: Record<string, string> = {
      'shap-waterfall.png': 'https://shap.readthedocs.io/en/latest/_images/example_notebooks_api_examples_plots_waterfall_3_0.png',
      'shap-capital-gain.png': 'https://shap.readthedocs.io/en/latest/_images/example_notebooks_api_examples_plots_waterfall_7_0.png',
      'air-bench-pipeline.svg': 'https://arxiv.org/html/2407.17436v1/air2024_pipeline.svg'
    };
    expect(cases[name]).toBeDefined();
    const shap = name.startsWith('shap-');
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.source_url).toBe(shap ? 'https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html' : 'https://arxiv.org/html/2407.17436v1#S3.F3');
    expect(entry.source_image_url).toBe(cases[name]);
    expect(entry.source_asset).toBe(prefix + name);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.sha256);
    expect(entry.source_page_snapshot).toBe(prefix + (shap ? 'shap-waterfall.html' : 'air-bench.html'));
    expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    const page = readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8');
    expect(page).toContain(cases[name].split('/').at(-1)!);
    expect(entry.license_evidence).toBe(prefix + (shap ? 'shap-LICENSE' : 'air-bench.html'));
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(entry.license).toBe(shap ? 'MIT' : 'CC BY 4.0');
    expect(entry.license_url).toBe(shap ? 'https://github.com/shap/shap/blob/cc2007eedf365e66f4cd7e5d37edef6d463b322b/LICENSE' : 'https://creativecommons.org/licenses/by/4.0/');
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).toContain(shap ? 'Copyright (c) 2018 Scott Lundberg' : 'License: CC BY 4.0');
    expect(entry.derivation).toBe(shap ? 'source-raster-copy' : 'source-svg-copy');
    expect(entry.provenance_confirmation).toBe(shap ? 'official-shap-documentation-asset' : 'versioned-arxiv-html-asset');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.crop_box_px).toBeUndefined();
    if (shap) {
      expect(entry.commit).toBe('cc2007eedf365e66f4cd7e5d37edef6d463b322b');
      expect(entry.source_code).toBe(prefix + 'shap-waterfall.ipynb');
      expect(sha256(resolve(repositoryRoot, entry.source_code!))).toBe(entry.source_code_sha256);
    }
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/imbalance-review-2026-10/')) {
    const prefix = '05 Источники/Originals/imbalance-2026-10-20/';
    const imPin = '8504e95f0160f61d1b617ca66f779646d2ee609e';
    const skPin = '866c0f51e7560ef0303cbcc5f159df5382ea9e3f';
    const expected: Record<string, [string, string, string]> = {
      'smote-generation.png': ['smote-generation.html', 'smote-generation-source.py', 'https://imbalanced-learn.org/0.14/_images/sphx_glr_plot_illustration_generation_sample_001.png'],
      'oversampling-comparison.png': ['over-gallery.html', '05 Источники/imbalanced-learn/0.14.2/original/gallery/plot_comparison_over_sampling.py', 'https://imbalanced-learn.org/0.14/_images/sphx_glr_plot_comparison_over_sampling_004.png'],
      'smote-illposed.png': ['over-gallery.html', '05 Источники/imbalanced-learn/0.14.2/original/gallery/plot_comparison_over_sampling.py', 'https://imbalanced-learn.org/0.14/_images/sphx_glr_plot_comparison_over_sampling_006.png'],
      'tomek-removal.png': ['tomek-gallery.html', '05 Источники/imbalanced-learn/0.14.2/original/gallery/plot_illustration_tomek_links.py', 'https://imbalanced-learn.org/0.14/_images/sphx_glr_plot_illustration_tomek_links_002.png'],
      'nearmiss-1.png': ['nearmiss-gallery.html', 'nearmiss-source.py', 'https://imbalanced-learn.org/0.14/_images/sphx_glr_plot_illustration_nearmiss_001.png'],
      'nearmiss-2.png': ['nearmiss-gallery.html', 'nearmiss-source.py', 'https://imbalanced-learn.org/0.14/_images/sphx_glr_plot_illustration_nearmiss_002.png'],
      'balanced-bagging.png': ['ensemble-gallery.html', '05 Источники/imbalanced-learn/0.14.2/original/gallery/plot_comparison_ensemble_classifier.py', 'https://imbalanced-learn.org/0.14/_images/sphx_glr_plot_comparison_ensemble_classifier_002.png'],
      'threshold-tradeoff.png': ['threshold.html', 'cost-sensitive-source.py', 'https://scikit-learn.org/stable/_images/sphx_glr_plot_cost_sensitive_learning_002.png'],
      'precision-recall.png': ['precision-recall.html', 'precision-recall-source.py', 'https://scikit-learn.org/stable/_images/sphx_glr_plot_precision_recall_001.png'],
      'calibration.png': ['calibration.html', 'calibration-source.py', 'https://scikit-learn.org/stable/_images/sphx_glr_plot_compare_calibration_001.png'],
    };
    const name = entry.asset.split('/').at(-1)!;
    expect(expected[name]).toBeDefined();
    const [page, code, image] = expected[name];
    const im = image.includes('imbalanced-learn.org');
    const pages: Record<string, string> = {
      'smote-generation.html': 'auto_examples/over-sampling/plot_illustration_generation_sample.html',
      'over-gallery.html': 'auto_examples/over-sampling/plot_comparison_over_sampling.html',
      'tomek-gallery.html': 'auto_examples/under-sampling/plot_illustration_tomek_links.html',
      'nearmiss-gallery.html': 'auto_examples/under-sampling/plot_illustration_nearmiss.html',
      'ensemble-gallery.html': 'auto_examples/ensemble/plot_comparison_ensemble_classifier.html',
      'threshold.html': 'modules/classification_threshold.html',
      'precision-recall.html': 'auto_examples/model_selection/plot_precision_recall.html',
      'calibration.html': 'modules/calibration.html',
    };
    expect(entry.source_url).toBe((im ? 'https://imbalanced-learn.org/0.14/' : 'https://scikit-learn.org/stable/') + pages[page]);
    const licensePath = im ? '05 Источники/imbalanced-learn/0.14.2/original/LICENSE' : prefix + 'sklearn-COPYING';
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.source_asset).toBe(prefix + name);
    expect(entry.source_image_url).toBe(image);
    expect(entry.source_page_snapshot).toBe(prefix + page);
    expect(entry.source_code).toBe(code.includes('/') ? code : prefix + code);
    expect(entry.commit).toBe(im ? imPin : skPin);
    expect(entry.license).toBe(im ? 'MIT' : 'BSD-3-Clause');
    expect(entry.license_url).toBe(im
      ? 'https://github.com/scikit-learn-contrib/imbalanced-learn/blob/' + imPin + '/LICENSE'
      : 'https://github.com/scikit-learn/scikit-learn/blob/' + skPin + '/COPYING');
    expect(entry.license_evidence).toBe(licensePath);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.provenance_confirmation).toBe(im ? 'official-imblearn-gallery-asset' : 'official-sklearn-gallery-asset');
    expect(entry.source_page).toBeUndefined();
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.render_page_size_px).toBeUndefined();
    for (const [path, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_code, entry.source_code_sha256], [entry.source_page_snapshot, entry.source_page_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]]) {
      expect(hash).toMatch(/^[a-f0-9]{64}$/);
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    }
    expect(entry.sha256).toBe(entry.source_asset_sha256);
    const html = readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8');
    // Sphinx snapshots do not necessarily contain their own absolute URL.
    // Check the exact declared endpoint above, then local page/figure identity.
    expect(html.includes(pages[page].split('/').at(-1)!.replace('.html', ''))).toBe(true);
    expect(html.includes(image.split('/').at(-1)!)).toBe(true);
    expect(readFileSync(resolve(repositoryRoot, licensePath), 'utf8')).toContain(im ? 'The MIT License' : 'BSD 3-Clause License');
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot, entry.asset)))[0]).toBeGreaterThanOrEqual(600);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/evaluation-review-2026-10/')) {
    const prefix = '05 Источники/Originals/evaluation-2026-10-20/';
    const pin = '1bade87e4c4e41d80d5f72ba2d7b38ab3194f044';
    const name = entry.asset.split('/').at(-1)!;
    const animated = ['ppl_full.gif', 'ppl_chunked.gif', 'ppl_sliding.gif'].includes(name);
    const papers: Record<string, [string, string, number, number[]]> = {
      'oren-full-figure1.png': ['2310.17623v2', 'oren', 2, [780, 550, 3880, 1370]],
      'ifeval-instructions-table1.png': ['2311.07911v1', 'ifeval', 3, [790, 590, 3890, 4790]],
    };
    expect(animated || name in papers).toBe(true);
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.license).toBe(animated ? 'CC BY-NC-SA 4.0' : 'CC BY 4.0');
    expect(entry.license_url).toBe(`https://creativecommons.org/licenses/${animated ? 'by-nc-sa' : 'by'}/4.0/`);
    for (const field of [entry.sha256, entry.source_asset_sha256, entry.license_evidence_sha256])
      expect(field).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    if (animated) {
      expect(entry.commit).toBe(pin);
      expect(entry.source_url).toBe(`https://huggingface.co/datasets/huggingface/documentation-images/blob/${pin}/${name}`);
      expect(entry.source_image_url).toBe(`https://huggingface.co/datasets/huggingface/documentation-images/resolve/${pin}/${name}`);
      expect(entry.source_asset).toBe(prefix + name);
      expect(entry.license_evidence).toBe(prefix + 'hf-images-README.md');
      expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).toContain('license: cc-by-nc-sa-4.0');
      expect(entry.derivation).toBe('source-raster-copy');
      expect(entry.provenance_confirmation).toBe('pinned-huggingface-source-asset');
      expect(entry.source_asset_sha256).toBe(entry.sha256);
      expect(entry.source_page).toBeUndefined();
      expect(entry.crop_box_px).toBeUndefined();
      expect(entry.render_page_size_px).toBeUndefined();
    } else {
      const [paper, local, page, crop] = papers[name];
      expect(entry.source_url).toBe(`https://arxiv.org/pdf/${paper}#page=${page}`);
      expect(entry.source_asset).toBe(prefix + local + '.pdf');
      expect(entry.license_evidence).toBe(prefix + local + '-arxiv.html');
      expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).toContain('creativecommons.org/licenses/by/4.0/');
      expect(entry.derivation).toBe('pdf-page-render-crop');
      expect(entry.provenance_confirmation).toBe('licensed-paper-figure');
      expect(entry.source_page).toBe(page);
      expect(entry.crop_box_px).toEqual(crop);
      expect(entry.render_page_size_px).toEqual([4637, 6000]);
      expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot, entry.asset))))
        .toEqual([crop[2] - crop[0], crop[3] - crop[1]]);
    }
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/edge-review-2026-10/')) {
    const prefix = '05 Источники/Originals/edge-2026-10-20/';
    const pin = '45ecc8d82fcae70c149cdce550d3b3d3411df913';
    const litert = 'https://developers.google.com/edge/litert/conversion/tensorflow/build/ondevice_training';
    const harvard = `https://github.com/harvard-edge/cs249r_book/blob/${pin}/book/quarto/contents/vol2/edge_intelligence/images/svg/`;
    const expected: Record<string, [string, string, string, number?, number[]?]> = {
      'tinytl-memory-figure1.png': ['https://arxiv.org/pdf/2007.11622v5#page=2', 'tinytl.pdf', 'tinytl-arxiv.html', 2, [800, 490, 3860, 1530]],
      'tinytl-activation-figure2.png': ['https://arxiv.org/pdf/2007.11622v5#page=4', 'tinytl.pdf', 'tinytl-arxiv.html', 4, [1090, 500, 3610, 2050]],
      'fedavg-initialization-figure1.png': ['https://proceedings.mlr.press/v54/mcmahan17a.html', 'fedavg.pdf', 'pmlr-license.html', 4, [2430, 505, 4330, 1415]],
      'litert-checkpoint.png': [litert, 'litert-checkpoint.png', 'litert-training.html'],
      'harvard-stragglers.svg': [harvard + 'edge_intelligence_straggler_cutoff_strip.svg', 'harvard-stragglers.svg', 'harvard-LICENSE.md'],
      'harvard-secure-agg.svg': [harvard + 'secure-agg.svg', 'harvard-secure-agg.svg', 'harvard-LICENSE.md'],
    };
    const name = entry.asset.split('/').at(-1)!;
    expect(expected[name]).toBeDefined();
    const [url, sourceName, proofName, page, crop] = expected[name];
    const svg = name.endsWith('.svg');
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.source_url).toBe(url);
    expect(entry.source_asset).toBe(prefix + sourceName);
    expect(entry.license_evidence).toBe(prefix + proofName);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.license).toBe(svg ? 'CC BY-NC-SA 4.0' : 'CC BY 4.0');
    const slug = svg ? 'by-nc-sa' : 'by';
    expect(entry.license_url).toBe(`https://creativecommons.org/licenses/${slug}/4.0/`);
    for (const field of [entry.sha256, entry.source_asset_sha256, entry.license_evidence_sha256])
      expect(field).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain(`creativecommons.org/licenses/${slug}/4.0/`);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    if (page) {
      expect(entry.derivation).toBe('pdf-page-render-crop');
      expect(entry.provenance_confirmation).toBe('licensed-paper-figure');
      expect(entry.source_page).toBe(page);
      expect(entry.crop_box_px).toEqual(crop);
      expect(entry.render_page_size_px).toEqual([4637, 6000]);
      expect(readFileSync(resolve(repositoryRoot, entry.source_asset!)).subarray(0, 5).toString()).toBe('%PDF-');
      expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot, entry.asset))))
        .toEqual([crop![2] - crop![0], crop![3] - crop![1]]);
    } else {
      expect(entry.derivation).toBe(svg ? 'source-svg-copy' : 'source-raster-copy');
      expect(entry.source_asset_sha256).toBe(entry.sha256);
      expect(entry.crop_box_px).toBeUndefined();
      expect(entry.source_page).toBeUndefined();
      expect(entry.render_page_size_px).toBeUndefined();
      expect(entry.provenance_confirmation).toBe(svg ? 'pinned-harvard-source-asset' : 'official-google-source-asset');
      if (svg) {
        expect(entry.commit).toBe(pin);
        expect(entry.source_image_url).toBe(url.replace('https://github.com/harvard-edge/cs249r_book/blob/', 'https://raw.githubusercontent.com/harvard-edge/cs249r_book/'));
      } else {
        expect(entry.source_image_url).toBe('https://developers.google.com/static/edge/litert/conversion/tensorflow/build/ondevice_training_files/output_fjiUbx7zIoLq_0.png');
      }
    }
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/resolution-2026-11-09/llava-onevision-anyres.png') {
    const source = '05 Источники/Originals/resolution-2026-10-20/onevision-v1-higher_res.png';
    const abstract = '05 Источники/Originals/resolution-2026-10-20/onevision-v1-abstract.html';
    const hash = 'db33ded7e7e6fb072235d58b79df115555753d29606ede070df2ac0365c79b1f';
    expect(entry).toMatchObject({
      author: 'Bo Li et al.', source_url: 'https://arxiv.org/html/2408.03326v1#S3.F2', source_image_url: 'https://arxiv.org/html/2408.03326v1/higher_res.png',
      source_figure_label: 'Figure 2, both panels', source_asset: source, source_asset_sha256: hash,
      sha256: hash, bytes: 361765, dimensions_px: [1899, 1319], license: 'CC BY-NC-ND 4.0',
      license_url: 'https://creativecommons.org/licenses/by-nc-nd/4.0/',
      license_evidence: abstract, license_evidence_sha256: 'e57b651eb1e0d03510ed23de758743cdfc3f8c4ce9d8746a397d45bb8495785d',
      derivation: 'source-raster-copy', rights_status: 'source-license-verified',
      provenance_confirmation: 'licensed-paper-figure', metadata_status: 'verified',
      used_in: ['00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции.md', 'en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions.md'],
    });
    expect(entry.crop_box_px).toBeUndefined();
    const path = resolve(repositoryRoot, entry.asset);
    const png = readFileSync(path);
    expect(sha256(path)).toBe(hash);
    expect(validateRenderedPngBytes(png)).toEqual([1899, 1319]);
    expect(png.length).toBe(361765);
    if (existsSync(resolve(repositoryRoot, source)))
      expect(png.equals(readFileSync(resolve(repositoryRoot, source)))).toBe(true);
    if (existsSync(resolve(repositoryRoot, abstract)))
      expect(sha256(resolve(repositoryRoot, abstract))).toBe(entry.license_evidence_sha256);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/vlm-training-2026-11-09/')) {
    const figures = {
      'prismatic-figure4.png': { number: 4, hash: '49aca74ebdc08c0627538ce329c7bad8fc32e280cfd64691095126852bcdcbe2', crop: [270, 310, 2718, 1165] },
      'prismatic-figure5.png': { number: 5, hash: 'aa5f932da53815588fae3af294f178148e3c321ac45969424c7639d9fb4decfb', crop: [351, 1465, 1449, 2465] },
    };
    const spec = figures[entry.asset.split('/').at(-1)! as keyof typeof figures];
    expect(spec).toBeDefined();
    const source = '05 Источники/Originals/vlm-training-2026-10-20/karamcheti24a.pdf';
    const license = '05 Источники/Originals/edge-2026-10-20/pmlr-license.html';
    expect(entry).toMatchObject({
      author: 'Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, Dorsa Sadigh',
      source_url: 'https://proceedings.mlr.press/v235/karamcheti24a.html',
      source_image_url: 'https://raw.githubusercontent.com/mlresearch/v235/main/assets/karamcheti24a/karamcheti24a.pdf#page=5',
      source_figure_label: `Figure ${spec.number}, PDF p. 5`,
      source_asset: source, source_asset_sha256: '5b19876fd49109700321a90abf00025227b17d82131d052919645ea7b2245003', sha256: spec.hash,
      license: 'CC BY 4.0', license_url: 'https://creativecommons.org/licenses/by/4.0/',
      license_evidence: license, license_evidence_sha256: '1c8123a6c840ca2752e383f0924fa23ba38552140a2cc54f69ea9df5acfcdd3d',
      derivation: 'pdf-page-render-crop', rights_status: 'source-license-verified',
      provenance_confirmation: 'licensed-paper-figure', metadata_status: 'verified',
      source_page: 5, render_dpi: 360, render_page_size_px: [3060, 3960], crop_box_px: spec.crop,
      used_in: ['00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные.md', 'en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data.md'],
    });
    const path = resolve(repositoryRoot, entry.asset);
    const [x0, y0, x1, y1] = spec.crop;
    expect(sha256(path)).toBe(spec.hash);
    expect(validateRenderedPngBytes(readFileSync(path))).toEqual([x1 - x0, y1 - y0]);
    for (const [path, hash] of [[source, entry.source_asset_sha256], [license, entry.license_evidence_sha256]]) {
      const file = resolve(repositoryRoot, path);
      if (existsSync(file)) expect(sha256(file)).toBe(hash);
    }
    return;
  }
  const connectorPaper = entry.asset === '00 Учебник/Assets/Figures/curated/connectors-2026-11-09/llava-figure1-projector.png';
  if (connectorPaper || entry.asset.startsWith('00 Учебник/Assets/Figures/curated/vision-objectives-2026-11-09/')) {
    const figures = {
      'llava-figure1-projector.png': {
        author: 'Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee',
        paper: 'https://arxiv.org/abs/2304.08485v2',
        figure: 'https://arxiv.org/pdf/2304.08485v2#page=4',
        pdf: '05 Источники/Originals/connectors-2026-10-20/llava.pdf',
        pdfHash: 'dc8bace378a282ea78caa9b5959306202fbfa3b8cedaf8a66541a9af18f4d3bf',
        pngHash: '310746447f5ad4ae00a2e54407bed5845795221ca7b863779edacb3b86d80f44',
        evidence: '05 Источники/Originals/connectors-2026-10-20/llava-v2-abstract.html',
        evidenceHash: 'ca833da8c03d0b5c3d8be5bb004e9e49c94f4ac8f5161efdb1297f68fa5f903c',
        page: 4, crop: [804, 1416, 2258, 1912],
      },
      'clip-figure1-training-and-zero-shot.png': {
        author: 'Alec Radford et al.',
        paper: 'https://proceedings.mlr.press/v139/radford21a.html',
        figure: 'https://proceedings.mlr.press/v139/radford21a/radford21a.pdf#page=2',
        pdf: '05 Источники/Originals/vision-encoders-2026-10-20/clip.pdf',
        pdfHash: '2a4952960f350e50a3a4a847371b80b8aaa09ba82d1c2a8f6b30450e392a1fe4',
        pngHash: '351759e7ee8bd134f865e7e9575af9f68a7f9c8b9bb9beb2fc1279c49db5419a',
        evidence: '05 Источники/Originals/edge-2026-10-20/pmlr-license.html',
        evidenceHash: '1c8123a6c840ca2752e383f0924fa23ba38552140a2cc54f69ea9df5acfcdd3d',
        page: 2, crop: [265, 324, 2725, 1213],
      },
      'siglip-parallelism.png': {
        author: 'Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer',
        paper: 'https://arxiv.org/abs/2303.15343v4',
        figure: 'https://arxiv.org/pdf/2303.15343v4#page=3',
        pdf: '05 Источники/Originals/vision-encoders-2026-10-20/siglip.pdf',
        pdfHash: '2efade8de3baeffe9eba759aab9f510ea8f8ac3c365f02f0da75d9baf9ffa2db',
        pngHash: '6e373e66257439d1ad7adb36b6b71c77d55b7f6562f620c3f41953873e679aa1',
        evidence: '05 Источники/Originals/vision-encoders-2026-10-20/siglip-abstract.html',
        evidenceHash: 'bb4246ec588170bc529126b69c34c98320cf1ca31109db74433e4401cf3fdcb7',
        page: 3, crop: [235, 342, 2745, 1461],
      },
    };
    const name = entry.asset.split('/').at(-1)!;
    const spec = figures[name as keyof typeof figures];
    expect(spec).toBeDefined();
    expect(entry).toMatchObject({
      author: spec.author, source_url: spec.paper, source_image_url: spec.figure,
      source_asset: spec.pdf, source_asset_sha256: spec.pdfHash, sha256: spec.pngHash,
      license_evidence: spec.evidence, license_evidence_sha256: spec.evidenceHash,
      license: 'CC BY 4.0', license_url: 'https://creativecommons.org/licenses/by/4.0/',
      rights_status: 'source-license-verified', metadata_status: 'verified',
      provenance_confirmation: 'licensed-paper-figure', derivation: 'pdf-page-render-crop',
      source_figure_label: `Figure 1, PDF p. ${spec.page}`,
      source_page: spec.page, render_dpi: 360, render_page_size_px: [3060, 3960],
      crop_box_px: spec.crop,
      used_in: connectorPaper ? ['00 Учебник/16 Multimodal Models/64a Connectors и fusion.md',
        'en/00 Textbook/16 Multimodal Models/64a Connectors and fusion.md'] : ['00 Учебник/16 Multimodal Models/64 Мультимодальные модели.md',
        'en/00 Textbook/16 Multimodal Models/64 Multimodal models.md'],
    });
    const image = resolve(repositoryRoot, entry.asset);
    const [x0, y0, x1, y1] = spec.crop;
    expect(sha256(image)).toBe(spec.pngHash);
    expect(validateRenderedPngBytes(readFileSync(image))).toEqual([x1 - x0, y1 - y0]);
    for (const [path, digest] of [[spec.pdf, spec.pdfHash], [spec.evidence, spec.evidenceHash]]) {
      const original = resolve(repositoryRoot, path);
      if (existsSync(original)) expect(sha256(original)).toBe(digest);
    }
    return;
  }
  const smoothingFigure = entry.asset.startsWith('00 Учебник/Assets/Figures/curated/robustness-smoothing-2026-11-09/');
  const genderShadesFigure = entry.asset === '00 Учебник/Assets/Figures/curated/responsible-audit-2026-11-09/gender-shades-figure3.png';
  if (smoothingFigure || genderShadesFigure
    || entry.asset === '00 Учебник/Assets/Figures/curated/fedbuff-2026-11-09/fedbuff-concurrency-figure2.png'
    || entry.asset === '00 Учебник/Assets/Figures/curated/privacy-audit-2026-11-09/rmia-low-fpr-figure1.png') {
    const smoothing: Record<string, {page: number; crop: number[]; pngHash: string}> = {
      'smoothing-votes-figure1.png': {page: 1, crop: [2305, 1300, 4135, 2345],
        pngHash: '08827e2c532a088188b2e79d63707212155d7be685e88b47fa14882d2d76b5e2'},
      'smoothing-halfspace-figure3.png': {page: 5, crop: [430, 500, 2190, 1375],
        pngHash: 'fa280ef5297f4e3151189001eb4932651800838a183634f263936b988813c17b'},
    };
    const spec = smoothingFigure ? {
      author: 'Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter',
      url: 'https://proceedings.mlr.press/v97/cohen19c.html',
      pdf: '05 Источники/Originals/robustness-smoothing-2026-11-09/cohen19c.pdf',
      pdfHash: '409715d2ae1cd52d39b0a78ed1227edbdaa575be76dd40ab66b9869e97de4551',
      ...smoothing[entry.asset.split('/').at(-1)!],
    } : entry.asset.endsWith('fedbuff-concurrency-figure2.png') ? {
      author: 'John Nguyen, Kshitiz Malik, Hongyuan Zhan, Ashkan Yousefpour, Mike Rabbat, Mani Malek, Dzmitry Huba',
      url: 'https://proceedings.mlr.press/v151/nguyen22b.html',
      pdf: '05 Источники/Originals/fedbuff-2026-11-09/nguyen22b.pdf',
      pdfHash: '3c2764067cecb4727fc33dc9de9c3a73f31d2794cb7f557457506faf470a474b',
      pngHash: 'f9457ab23f9d443a7b4805da9e0cc278ab2d0a1c48cb2139f6b292aa5ce8333b',
      page: 3, crop: [455, 2090, 2275, 2840],
    } : genderShadesFigure ? {
      author: 'Joy Buolamwini, Timnit Gebru',
      url: 'https://proceedings.mlr.press/v81/buolamwini18a.html',
      pdf: '05 Источники/Originals/responsible-2026-10-20/buolamwini18a.pdf',
      pdfHash: 'e9e6d944715d14b03c383b1e1e2f4f7aa7ffaa9bd610708bd06645f03c64e7fa',
      pngHash: '3c0aed571f22a84f1c1171acf44a46fdf56636855af75fcbb094c1e28821543f',
      page: 6, crop: [1544, 445, 2621, 825],
    } : {
      author: 'Sajjad Zarifzadeh, Philippe Liu, Reza Shokri',
      url: 'https://proceedings.mlr.press/v235/zarifzadeh24a.html',
      pdf: '05 Источники/Originals/privacy-audit-2026-11-09/zarifzadeh24a.pdf',
      pdfHash: 'd026d8b77211e614d6ea803fb3908f2890daf64835b03b8ce89a969d4a8e6c87',
      pngHash: '518fa5500bd0d38d58d5b595a408bf23ea6a46908adc0cc0d7162e48a77e9646',
      page: 2, crop: [665, 480, 1940, 1650],
    };
    expect(entry.author).toBe(spec.author);
    expect(entry.source_url).toBe(spec.url);
    expect(entry.source_asset).toBe(spec.pdf);
    expect(entry.source_asset_sha256).toBe(spec.pdfHash);
    expect(entry.license_evidence).toBe('05 Источники/Originals/edge-2026-10-20/pmlr-license.html');
    expect(entry.license_evidence_sha256).toBe('1c8123a6c840ca2752e383f0924fa23ba38552140a2cc54f69ea9df5acfcdd3d');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.provenance_confirmation).toBe('licensed-paper-figure');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.source_page).toBe(spec.page);
    expect(entry.crop_box_px).toEqual(spec.crop);
    expect(entry.render_page_size_px).toEqual(genderShadesFigure ? [3060, 3960] : [4637, 6000]);
    if (genderShadesFigure) {
      expect(entry.render_dpi).toBe(360);
      expect(entry.source_image_url).toBe('https://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf#page=6');
      expect(entry.rights_status).toBe('source-license-verified');
    }
    const image = resolve(repositoryRoot, entry.asset);
    expect(sha256(image)).toBe(spec.pngHash);
    expect(sha256(image)).toBe(entry.sha256);
    const [x0, y0, x1, y1] = spec.crop;
    expect(validateRenderedPngBytes(readFileSync(image))).toEqual([x1 - x0, y1 - y0]);
    // The local checker also re-renders the original; the archive is not a
    // prerequisite for checking this image and its provenance in a checkout.
    const source = resolve(repositoryRoot, entry.source_asset!);
    if (existsSync(source)) expect(sha256(source)).toBe(entry.source_asset_sha256);
    const agreement = resolve(repositoryRoot, entry.license_evidence!);
    if (existsSync(agreement)) expect(sha256(agreement)).toBe(entry.license_evidence_sha256);
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/mlops-review-2026-10/')) {
    const prefix = '05 Источники/Originals/mlops-2026-10-20/';
    const name = entry.asset.split('/').at(-1)!;
    const cloud = 'https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning';
    const cloudImages = 'https://docs.cloud.google.com/static/architecture/images/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-';
    const pin = '45ecc8d82fcae70c149cdce550d3b3d3411df913';
    const expected: Record<string, [string, string, string, string]> = {
      'google-manual.svg': [cloud, cloudImages + '2-manual-ml.svg', 'google-mlops.html', 'CC BY 4.0'],
      'google-ct.svg': [cloud, cloudImages + '3-ml-automation-ct.svg', 'google-mlops.html', 'CC BY 4.0'],
      'google-cicd-stages.svg': [cloud, cloudImages + '5-stages.svg', 'google-mlops.html', 'CC BY 4.0'],
      'sre-canary-aggregate.webp': ['https://sre.google/workbook/canarying-releases/', 'https://sre.google/img/dynamic/sre-workbook/c7c0b53f6b669e88be6f9b652f6262e9.webp', 'google-canary.html', 'CC BY-NC-ND 4.0'],
      'sre-canary-versions.webp': ['https://sre.google/workbook/canarying-releases/', 'https://sre.google/img/dynamic/sre-workbook/aa048da9e213ce8192db45d653eb228f.webp', 'google-canary.html', 'CC BY-NC-ND 4.0'],
      'sre-alert-windows.webp': ['https://sre.google/workbook/alerting-on-slos/', 'https://sre.google/img/dynamic/sre-workbook/8e0f30e8897c5d378ee005a87e074641.webp', 'google-slo-alerting.html', 'CC BY-NC-ND 4.0'],
      'harvard-dependency.svg': [`https://github.com/harvard-edge/cs249r_book/blob/${pin}/book/quarto/contents/vol2/ops_scale/images/svg/ops_scale_cross_model_blast.svg`, `https://raw.githubusercontent.com/harvard-edge/cs249r_book/${pin}/book/quarto/contents/vol2/ops_scale/images/svg/ops_scale_cross_model_blast.svg`, 'harvard-LICENSE.md', 'CC BY-NC-SA 4.0']
    };
    expect(expected[name]).toBeDefined();
    const [page, imageUrl, evidence, license] = expected[name];
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.source_url).toBe(page);
    expect(entry.source_image_url).toBe(imageUrl);
    expect(entry.source_asset).toBe(prefix + name);
    expect(entry.license_evidence).toBe(prefix + evidence);
    expect(entry.license).toBe(license);
    const slug = license === 'CC BY 4.0' ? 'by' : license === 'CC BY-NC-ND 4.0' ? 'by-nc-nd' : 'by-nc-sa';
    expect(entry.license_url).toBe(`https://creativecommons.org/licenses/${slug}/4.0/`);
    expect(entry.metadata_status).toBe('verified');
    const svg = name.endsWith('.svg');
    expect(entry.derivation).toBe(svg ? 'source-svg-copy' : 'source-raster-copy');
    expect(entry.provenance_confirmation).toBe(name.startsWith('harvard') ? 'pinned-harvard-source-asset' : 'official-google-source-asset');
    for (const hash of [entry.sha256, entry.source_asset_sha256, entry.license_evidence_sha256]) {
      expect(hash).toMatch(/^[a-f0-9]{64}$/);
    }
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    const proof = readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8');
    expect(proof).toContain(`creativecommons.org/licenses/${slug}/4.0/`);
    if (name.startsWith('harvard')) {
      expect(entry.commit).toBe(pin);
      expect(readFileSync(resolve(repositoryRoot, prefix + 'harvard-ops-scale.qmd'), 'utf8')).toContain('images/svg/ops_scale_cross_model_blast.svg');
    } else {
      expect(proof).toContain(imageUrl.replace('https://docs.cloud.google.com', '').replace('https://sre.google', ''));
    }
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.render_page_size_px).toBeUndefined();
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/ml-workflow-review-2026-10/')) {
    const prefix = '05 Источники/Originals/ml-workflow-2026-10-20/';
    const names: Record<string, string> = {
      'google-pipes.png': 'pipes.png',
      'google-data-process.png': 'data-process.png',
      'google-train.png': 'train.png',
      'google-offline-online.png': 'offline-online.png',
      'google-prediction-process.png': 'prediction-process.png',
      'google-prediction-log.png': 'prediction-log.png'
    };
    const name = entry.asset.split('/').at(-1)!;
    expect(names[name]).toBeDefined();
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.source_url).toBe('https://developers.google.com/machine-learning/managing-ml-projects/pipelines');
    expect(entry.source_image_url).toBe('https://developers.google.com/static/machine-learning/managing-ml-projects/images/' + names[name]);
    expect(entry.source_asset).toBe(prefix + name);
    expect(entry.license_evidence).toBe(prefix + 'google-pipelines.html');
    expect(entry.license_evidence_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    const html = readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8');
    expect(html).toContain('https://creativecommons.org/licenses/by/4.0/');
    expect(html).toContain(entry.source_image_url!.replace('https://developers.google.com', ''));
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.provenance_confirmation).toBe('official-google-ml-course-asset');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.sha256);
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.render_page_size_px).toBeUndefined();
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/inference-queueing-review-2026-10/')) {
    const prefix = '05 Источники/Originals/inference-queueing-2026-10-20/';
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(entry.source_asset_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(entry.license_evidence_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    const evidence = readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8');
    const name = entry.asset.split('/').at(-1)!;
    if (name.startsWith('mit-')) {
      const cases: Record<string, { page: number; crop: number[]; size: number[] }> = {
        'mit-mg1-waiting-timeline.png': { page: 5, crop: [1580, 1520, 4300, 2380], size: [2720, 860] },
        'mit-mg1-residual-time.png': { page: 6, crop: [1380, 800, 3800, 1800], size: [2420, 1000] }
      };
      expect(cases[name]).toBeDefined();
      const expected = cases[name]!;
      expect(entry.source_url).toBe('https://ocw.mit.edu/courses/6-263j-data-communication-networks-fall-2002/65d9ab519ec4851af812f4b89ffaeedc_Lectures8_9.pdf#page=' + expected.page);
      expect(entry.source_asset).toBe('05 Источники/Courses/MIT 6.263J/Lectures/2002-lectures8-9.pdf');
      expect(entry.license_evidence).toBe(prefix + 'mit-lecture-resource.html');
      expect(evidence).toContain('65d9ab519ec4851af812f4b89ffaeedc_Lectures8_9.pdf');
      expect(evidence).toContain('https://creativecommons.org/licenses/by-nc-sa/4.0/');
      expect(entry.license).toBe('CC BY-NC-SA 4.0');
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-nc-sa/4.0/');
      expect(entry.provenance_confirmation).toBe('source-license-verified');
      expect(entry.derivation).toBe('pdf-page-render-crop');
      expect(entry.source_page).toBe(expected.page);
      expect(entry.render_page_size_px).toEqual([6000, 4637]);
      expect(entry.crop_box_px).toEqual(expected.crop);
      expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot, entry.asset)))).toEqual(expected.size);
    } else {
      const cases: Record<string, { hash: string; page: string; anchor: string }> = {
        'sre-normal-load.webp': { hash: 'd1a1ed1e7168236f34825cb718beafbd', page: 'addressing-cascading-failures', anchor: 'server-overload' },
        'sre-failover-load.webp': { hash: '1aa29313a813547ca9536349c8bd6f51', page: 'addressing-cascading-failures', anchor: 'server-overload' },
        'sre-retry-stack.webp': { hash: '0c4d69c7d44b3e0a40129c2e6bb021fb', page: 'handling-overload', anchor: 'deciding-to-retry' }
      };
      expect(cases[name]).toBeDefined();
      const expected = cases[name]!;
      expect(entry.source_url).toBe('https://sre.google/sre-book/' + expected.page + '/#' + expected.anchor);
      expect(entry.source_image_url).toBe('https://sre.google/img/dynamic/sre-book/' + expected.hash + '.webp');
      expect(entry.source_asset).toBe(prefix + name);
      expect(entry.license_evidence).toBe(prefix + (expected.page === 'handling-overload' ? 'sre-handling-overload.html' : 'sre-cascading-failures.html'));
      expect(evidence).toContain('/img/dynamic/sre-book/' + expected.hash + '.webp');
      expect(evidence).toContain('https://creativecommons.org/licenses/by-nc-nd/4.0/');
      expect(entry.license).toBe('CC BY-NC-ND 4.0');
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-nc-nd/4.0/');
      expect(entry.provenance_confirmation).toBe('official-sre-queueing-asset');
      expect(entry.derivation).toBe('source-raster-copy');
      expect(entry.source_asset_sha256).toBe(entry.sha256);
      expect(entry.crop_box_px).toBeUndefined();
      expect(entry.render_page_size_px).toBeUndefined();
    }
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.provenance_confirmation === 'official-sre-book-asset') {
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.source_url).toMatch(/^https:\/\/sre\.google\/workbook\/alerting-on-slos\/#(?:4-alert-on-burn-rate|6-multiwindow-multi-burn-rate-alerts)$/);
    expect(entry.source_image_url).toMatch(/^https:\/\/sre\.google\/img\/dynamic\/sre-workbook\/[a-f0-9]{32}\.webp$/);
    expect(entry.source_asset).toMatch(/^05 Источники\/Originals\/inference-benchmark-2026-10-20\/sre-(?:burn-budget|alert-windows)\.webp$/);
    expect(entry.license_evidence).toBe('05 Источники/Originals/inference-benchmark-2026-10-20/sre-alerting-on-slos.html');
    expect(entry.license_evidence_sha256).toMatch(/^[a-f0-9]{64}$/);
    const evidence = resolve(repositoryRoot, entry.license_evidence!);
    expect(sha256(evidence)).toBe(entry.license_evidence_sha256);
    const html = readFileSync(evidence, 'utf8');
    expect(html).toContain('https://creativecommons.org/licenses/by-nc-nd/4.0/');
    expect(html).toContain(entry.source_image_url!.replace('https://sre.google', ''));
    expect(entry.license).toBe('CC BY-NC-ND 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-nc-nd/4.0/');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.render_page_size_px).toBeUndefined();
    expect(entry.asset).toMatch(/^00 Учебник\/Assets\/Figures\/curated\/inference-benchmark-review-2026-10\/sre-(?:burn-budget|alert-windows)\.webp$/);
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(entry.sha256).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/inference-benchmark-review-2026-10/distserve-figure1-original.png') {
    // A re-extraction of an existing approved figure is not new CC clearance.
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.license).toBe('Existing reproduction permission record; no new named license asserted');
    expect(entry.license_url).toBe('https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf');
    expect(entry.provenance_confirmation).toBe('user-confirmed-open-materials');
    expect(entry.metadata_status).toBe('needs-confirmation');
    expect(entry.rights_status).toBe('existing-figure-reextraction; rights-not-reaudited');
    expect(entry.source_url).toBe('https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=2');
    expect(entry.source_asset).toBe('05 Источники/Papers/Source PDFs/distserve-osdi24-zhong-yinmin.pdf');
    expect(entry.source_asset_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(entry.derivation).toBe('pdf-page-render-crop');
    expect(entry.source_page).toBe(2);
    expect(entry.render_page_size_px).toEqual([4637, 6000]);
    expect(entry.crop_box_px).toEqual([2380, 1920, 4270, 3780]);
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot, entry.asset)))).toEqual([1890, 1860]);
    return;
  }
  if (entry.provenance_confirmation === 'pinned-arxiv-source-archive-member') {
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.source_url).toBe('https://arxiv.org/src/2407.08608v1');
    expect(entry.source_asset).toBe('raw/papers/flashattention-3-2407.08608v1/source.tar.gz');
    expect(entry.source_member).toBe('figs/pingpong_pipelining.png');
    expect(entry.source_asset_sha256).toBe('642f40096dcd70a0654a93798bd49dd1448e89504273910f525bf8a08be4f766');
    expect(entry.source_member_sha256).toBe('73c1561658aa3a2520367192045056ab72fe66082ea61ebe4f4a1202570b285c');
    expect(entry.sha256).toBe(entry.source_member_sha256);
    expect(entry.derivation).toBe('source-archive-member-copy');
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.asset).toMatch(curatedPngPath);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot, entry.asset)))).toEqual([3285, 671]);
    const archive = resolve(repositoryRoot, entry.source_asset!);
    // Immutable raw sources are optional in public CI, not in this local audit.
    if (existsSync(archive)) {
      expect(sha256(archive)).toBe(entry.source_asset_sha256);
      const member = execFileSync('tar', ['-xOf', archive, entry.source_member!], { maxBuffer: 16 * 1024 * 1024 });
      expect(createHash('sha256').update(member).digest('hex')).toBe(entry.sha256);
    }
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/fault-tolerance-2026-10-23/')) {
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/fault-tolerance-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    const spec = rows.find(row => row.asset === entry.asset);
    expect(spec).toBeTruthy();
    for (const [key, value] of Object.entries(spec!))
      expect((entry as unknown as Record<string, unknown>)[key]).toEqual(value);
    const cases: Record<string, [string, number, number]> = {
      'harvard-checkpoint-tradeoff.png': ['301fbccf3fa8902baef08bdb449dda820c8bcd589522da18634d116b2a7cbe80', 4739, 2932],
      'harvard-cluster-fragmentation.png': ['e5b8dbf23492b08aa30c679cd0e2bbfddb3093a35aed391e694f9b3186cb1945', 1783, 904]
    };
    const [digest, width, height] = cases[entry.asset.split('/').at(-1)!];
    expect(entry.sha256).toBe(digest);
    expect(entry.source_asset_sha256).toBe(digest);
    expect(entry.commit).toBe('45ecc8d82fcae70c149cdce550d3b3d3411df913');
    expect(entry.source_url).toContain('/blob/' + entry.commit + '/');
    expect(entry.source_image_url).toMatch(/^https:\/\/mlsysbook\.ai\/vol2\/.+_files\/figure-html\/cell-\d+-output-1\.png$/);
    expect(entry.license).toBe('CC BY-NC-SA 4.0');
    expect(entry.license_evidence_sha256).toBe('b6f4689d6fccf48d82d0871a3b994a122427fcb0b278971f6ba008be44e2c926');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.provenance_confirmation).toBe('rendered-source-snapshot-and-pinned-figure-code');
    expect(entry.modifications).toContain('Commit pins the QMD generating code, not the rendered image URL.');
    expect(entry.crop_box_px).toBeUndefined();
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([width, height]);
    for (const [path, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.asset))).toEqual(
      readFileSync(resolve(repositoryRoot, entry.source_asset!)));
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/network-checkpoint-2026-10-23/')) {
    const rows = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/network-checkpoint-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    const spec = rows.find(row => row.asset === entry.asset);
    expect(spec).toBeTruthy();
    for (const [key, value] of Object.entries(spec!))
      expect((entry as unknown as Record<string, unknown>)[key]).toEqual(value);
    const cases: Record<string, [string, string, string, number, number]> = {
      'jax-superpod-diagram.png': ['2817ab222ef378b67aa1fd909f8821032a9a58ee90b040060ec2f815c555c560',
        '3ae1dee9eef3430aa671162397afe5dd79f299ca', 'MIT', 1936, 862],
      'deepspeed-ucp-flowchart.png': ['c1c581ce327bafff6316cafb8012191e9d1c4b2b0dcf89eeb684d77fb2436ee0',
        'c2c81993948fc28385542196c8544fb442017987', 'Apache-2.0', 819, 279]
    };
    const [digest, pin, license, width, height] = cases[entry.asset.split('/').at(-1)!];
    expect(entry.sha256).toBe(digest);
    expect(entry.source_asset_sha256).toBe(digest);
    expect(entry.commit).toBe(pin);
    expect(entry.source_image_url).toContain('/' + pin + '/');
    expect(entry.license_url).toContain('/blob/' + pin + '/LICENSE');
    expect(entry.license).toBe(license);
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([width, height]);
    for (const [path, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.asset))).toEqual(
      readFileSync(resolve(repositoryRoot, entry.source_asset!)));
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain(license === 'MIT' ? 'Copyright (c) 2025 Google LLC' : 'Apache License');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/expert-hybrid-2026-10-23/')) {
    const [spec] = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/expert-hybrid-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    for (const [key, value] of Object.entries(spec))
      expect((entry as unknown as Record<string, unknown>)[key]).toEqual(value);
    const pin = 'c93890dd4b22356ee18ed9b863e6f2d0909729bc';
    expect(entry.sha256).toBe('bca083eab840eaa8e8c0ccfe0306b50a6bb38ebf42c85d67f4d80593f800c293');
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.commit).toBe(pin);
    expect(entry.source_image_url).toBe(
      `https://huggingface.co/spaces/nanotron/ultrascale-playbook/resolve/${pin}/assets/images/5d_nutshell_ep.svg`);
    expect(entry.source_url).toBe(entry.source_image_url!.replace('/resolve/', '/blob/'));
    expect(entry.license).toBe('Apache-2.0');
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.crop_box_px).toBeUndefined();
    for (const [path, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.asset), 'utf8'))
      .toContain('viewBox="0 0 2856.9763622634646 550.2008447424914"');
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).toContain('license: apache-2.0');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/fsdp-2026-10-23/')) {
    const specs = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/fsdp-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    const spec = specs.find(row => row.asset === entry.asset);
    expect(spec).toBeTruthy();
    for (const [key, value] of Object.entries(spec!))
      expect((entry as unknown as Record<string, unknown>)[key]).toEqual(value);
    const expected: Record<string, [string, string, string, number]> = {
      'hf-zero-memory.png': ['zero_memory.svg', 'c27c9cec9cac888ad998e07b3d2c5d6fd2fd8a7bc977ff255ccd101371dd022c', 'a882554481e76fe9909daa34e30430f8dd267d0a57957e1890ee5b3c3ac4f984', 1366],
      'hf-fsdp-forward.png': ['dp_zero3_fwd.svg', 'df0793b7d70e31ce289bf135288dc7a8c08f5888ef91cc4f9b1224c8248d9b3b', 'a5fec6e1f1ff02270a24c984d7591716834d2699cd7af4bb606ef3b8a5d5f92d', 889],
      'hf-fsdp-backward.png': ['dp_zero3_bwd.svg', 'ed98197af27915a3b34401939e9dc8d4fb99a218350dc255354ad89e10d6918a', '8ac566b6122542f899fd79ebda9fd3673fc32444d0e40fa27355c9800dfee30f', 989]
    };
    const [sourceName, sourceHash, outputHash, height] = expected[entry.asset.split('/').at(-1)!];
    const pin = 'c93890dd4b22356ee18ed9b863e6f2d0909729bc';
    expect(entry.sha256).toBe(outputHash);
    expect(entry.source_asset_sha256).toBe(sourceHash);
    expect(entry.source_code_sha256).toBe('867363e410dc9d7562a1b2d198f8a11d2f5a3003b7409aab4020297ae1a24947');
    expect(entry.commit).toBe(pin);
    expect(entry.source_image_url).toBe(`https://huggingface.co/spaces/nanotron/ultrascale-playbook/resolve/${pin}/assets/images/${sourceName}`);
    expect(entry.license).toBe('Apache-2.0');
    expect(entry.license_url).toBe(`https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/${pin}/README.md`);
    expect(entry.derivation).toBe('source-svg-rasterization');
    expect(entry.crop_box_px).toBeUndefined();
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([2400, height]);
    for (const [path, hash] of [[entry.asset, entry.sha256], [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256], [entry.source_code, entry.source_code_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).toContain('license: apache-2.0');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.derivation === 'source-svg-rasterization') {
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'rasterization description');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.commit).toBe(d2lCommit);
    expect(entry.source_url).toMatch(/^https:\/\/d2l\.ai\/(?:[\w-]+\/)+[\w-]+\.html(?:#[\w-]+)?$/);
    expect(entry.source_image_url).toMatch(new RegExp(
      `^https://raw\\.githubusercontent\\.com/d2l-ai/d2l-en/${d2lCommit}/img/[\\w-]+\\.svg$`
    ));
    expect(entry.license).toBe('CC BY-SA 4.0');
    expect(entry.license_url).toBe(`https://github.com/d2l-ai/d2l-en/blob/${d2lCommit}/LICENSE`);
    expect(entry.source_asset).toMatch(/^00 Учебник\/Assets\/Figures\/curated\/(?:[\w-]+\/)+[\w-]+-original\.svg$/);
    expect(entry.asset).toMatch(curatedPngPath);
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.render_page_size_px).toBeUndefined();
    expect(entry.source_asset_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    const source = resolve(repositoryRoot, entry.source_asset!);
    expect(sha256(source)).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expectDecodableImage(source);
    const svg = readFileSync(source, 'utf8');
    expect(svg.trimEnd()).toMatch(/<\/svg>$/);
    const viewBox = svg.match(/<svg\b[^>]*\bviewBox="([^"]+)"/i)?.[1].trim().split(/[\s,]+/).map(Number);
    expect(viewBox).toHaveLength(4);
    expect(viewBox!.every(Number.isFinite)).toBe(true);
    expect(viewBox![2]).toBeGreaterThan(0);
    expect(viewBox![3]).toBeGreaterThan(0);
    const [width, height] = validateRenderedPngBytes(readFileSync(resolve(repositoryRoot, entry.asset)));
    const expectedHeight = width * viewBox![3] / viewBox![2];
    expect(height).toBeGreaterThanOrEqual(Math.floor(expectedHeight));
    expect(height).toBeLessThanOrEqual(Math.ceil(expectedHeight));
    return;
  }

  if (entry.derivation === 'pdf-page-render-crop'
    && entry.provenance_confirmation === 'user-confirmed-open-materials'
    && (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/scaling-review-2026-10/')
      || entry.source_asset?.startsWith('05 Источники/'))) {
    // Existing local Stanford material, not a verified third-party license.
    // The narrower route must not relax checks for licensed paper crops.
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'crop description');
    expect(entry.metadata_status).toBe('needs-confirmation');
    expect(entry.license).toBe('User permission record (not a named license)');
    expect(entry.license_url).toBe('https://cs336.stanford.edu/');
    expect(entry.rights_status).toBe('existing-local-reformat; third-party-rights-unverified');
    expect(entry.commit).toBe(cs336Commit);
    const source = entry.source_asset?.match(/^05 Источники\/Courses\/Stanford CS336 Spring 2026\/Lectures\/repository\/(lecture_(?:09|11)\.pdf)$/);
    expect(source).toBeTruthy();
    expect(Number.isInteger(entry.source_page) && Number(entry.source_page) > 0).toBe(true);
    // Counts checked with pdfinfo for the two documents at cs336Commit.
    expect(entry.source_page).toBeLessThanOrEqual(source?.[1] === 'lecture_09.pdf' ? 57 : 58);
    expect(entry.source_url).toBe(`https://github.com/stanford-cs336/lectures/blob/${cs336Commit}/${source?.[1]}#page=${entry.source_page}`);
    expect(entry.source_asset_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    const sourcePath = resolve(repositoryRoot, entry.source_asset!);
    expect(sha256(sourcePath)).toBe(entry.source_asset_sha256);
    expect(readFileSync(sourcePath).subarray(0, 5).toString('ascii')).toBe('%PDF-');
    expect(entry.asset).toMatch(curatedPngPath);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(entry.render_page_size_px).toEqual([2400, 1352]);
    expect(Array.isArray(entry.crop_box_px)).toBe(true);
    const crop = entry.crop_box_px as number[];
    expect(crop).toHaveLength(4);
    expect(crop.every(Number.isInteger)).toBe(true);
    const [left, top, right, bottom] = crop;
    expect(left).toBeGreaterThanOrEqual(0);
    expect(top).toBeGreaterThanOrEqual(0);
    expect(right).toBeGreaterThan(left);
    expect(bottom).toBeGreaterThan(top);
    expect(right).toBeLessThanOrEqual(2400);
    expect(bottom).toBeLessThanOrEqual(1352);
    expect(validateRenderedPngBytes(readFileSync(resolve(repositoryRoot, entry.asset)))).toEqual([right - left, bottom - top]);
    return;
  }

  if (entry.provenance_confirmation === 'official-course-asset') {
    // MIT OCW publishes course images at named, hash-prefixed URLs, not Git blobs.
    // Keep the official course page, exact asset URL and unchanged file hash.
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.source_asset).toBeUndefined();
    expect(entry.source_url).toMatch(/^https:\/\/ocw\.mit\.edu\/courses\/6-004-computation-structures-spring-2017\/pages\/c(?:7|21)\/c(?:7|21)s1\/$/);
    expect(entry.source_image_url).toMatch(/^https:\/\/ocw\.mit\.edu\/courses\/6-004-computation-structures-spring-2017\/[a-f0-9]{32}_Slide\d+\.png$/);
    expect(entry.license).toBe('CC BY-NC-SA 4.0');
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/by-nc-sa/4.0/');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.asset).toMatch(/^00 Учебник\/Assets\/Figures\/curated\/(?:[\w-]+\/)+[\w-]+\.png$/);
    expect(entry.source_asset_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(entry.sha256).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (!entry.source_asset && entry.source_url?.startsWith('https://d2l.ai/')) {
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.license).toBe('CC BY-SA 4.0');
    expect(entry.license_url).toBe('https://github.com/d2l-ai/d2l-en/blob/master/LICENSE');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.source_image_url).toMatch(/^https:\/\/d2l\.ai\/_images\/output_[a-zA-Z0-9_-]+\.svg$/);
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(entry.license_evidence).toBe('05 Источники/Originals/probability-2026-10-21/d2l-LICENSE.txt');
    expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain('Creative Commons Attribution-ShareAlike 4.0 International Public License');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/zero-bubble-2026-10-23/')) {
    const prefix = '00 Учебник/Assets/Figures/curated/zero-bubble-2026-10-23/';
    const archive = '05 Источники/Originals/zero-bubble-2026-10-23/';
    const pin = '34d85332bccf8111db03eed694bbcd10f8231ee4';
    const page = 'https://huggingface.co/spaces/sail/zero-bubble-pipeline-parallellism/blob/' + pin + '/';
    const originals = {"dependency.png":["249d97877ac94a178b07524f927409e3e8f34fc1210ca94b25cd3d0870cad959",[1144,610],"8B9thyMiLgysNi_m_O3Qn.png"],"zb1p.png":["a3fafc7065c66003966de33383461f269cefc94d02ba46f21a1ee5e683636522",[1320,166],"EcTFvbjfM7soUXDYyn1Xu.png"],"zb2p.png":["8bbe08ba2f89d1a00c697494645d72c7ed2b51063085aeec77c2ab0368c17168",[1318,200],"8jFI_rO69BREKqiSFHIOL.png"],"zbv.png":["28a83d627002633f670d662903c50dd62e210febb9a98e5ed278ba2845afcd4f",[1318,352],"VRfjNVXakAU3MQK3h6OKa.png"],"optimizer.png":["a06373e7564166362da11363acab9fb4c40d517d8582e56144fe3a98031a3692",[1180,372],"hRPFqaFxJ20wm2omwyKmO.png"]} as Record<string, [string, number[], string]>;
    const original = originals[entry.asset.slice(prefix.length)];
    expect(original).toBeTruthy();
    const specs = JSON.parse(readFileSync(resolve(repositoryRoot, prefix + 'assets.json'), 'utf8')) as CuratedAsset[];
    const spec = specs.find(row => row.asset === entry.asset);
    expect(spec).toBeTruthy();
    for (const [key, value] of Object.entries(spec!))
      expect((entry as unknown as Record<string, unknown>)[key]).toEqual(value);
    expect(entry.commit).toBe(pin);
    expect(entry.sha256).toBe(original[0]);
    expect(entry.source_asset).toBe(entry.asset);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(entry.source_url).toBe(page + 'description2.md');
    expect(entry.source_image_url).toBe('https://cdn-uploads.huggingface.co/production/uploads/63510eea0b94548566dad923/' + original[2]);
    expect(entry.source_page_snapshot).toBe(archive + 'description2.md');
    expect(entry.source_page_sha256).toBe('fe5d715bc801d3a34af0a84ff6a8a4c5acd084c8ee92c6244fdd86494b87cef7');
    expect(entry.license).toBe('Apache-2.0');
    expect(entry.license_url).toBe(page + 'README.md');
    expect(entry.license_evidence).toBe(archive + 'README.md');
    expect(entry.license_evidence_sha256).toBe('ef973e0e1cde4d1ad1e1c8f26ef6e2360c8e7e6f9a99cc2be0f605d5f5e9f7cc');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.crop_box_px).toBeUndefined();
    for (const [path, hash] of [[entry.asset, entry.sha256],
      [entry.source_page_snapshot, entry.source_page_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8'))
      .toContain(entry.source_image_url);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain('license: apache-2.0');
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual(original[1]);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/pipeline-2026-10-23/')) {
    const [spec] = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/pipeline-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    for (const [key, value] of Object.entries(spec))
      expect((entry as unknown as Record<string, unknown>)[key]).toEqual(value);
    const pin = 'c93890dd4b22356ee18ed9b863e6f2d0909729bc';
    expect(entry.sha256).toBe('097161c8edc79166680595bfcb4538a5bda7ab4ce7693038129f7d86414229bd');
    expect(entry.commit).toBe(pin);
    expect(entry.source_image_url).toBe(`https://huggingface.co/spaces/nanotron/ultrascale-playbook/resolve/${pin}/assets/images/pp_afab.svg`);
    expect(entry.source_url).toBe(entry.source_image_url!.replace('/resolve/', '/blob/'));
    expect(entry.license).toBe('Apache-2.0');
    expect(entry.license_url).toBe(`https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/${pin}/README.md`);
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    for (const [path, hash] of [[entry.asset, entry.sha256],
      [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).toContain('license: apache-2.0');
    expect(readFileSync(resolve(repositoryRoot, entry.asset), 'utf8')).toContain('viewBox="0 0 945.96716 242.76773"');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/tensor-sequence-2026-10-23/')) {
    const specs = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/tensor-sequence-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    const spec = specs.find(row => row.asset === entry.asset);
    expect(spec).toBeTruthy();
    for (const [key, value] of Object.entries(spec!))
      expect((entry as unknown as Record<string, unknown>)[key]).toEqual(value);
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    if (entry.asset.endsWith('ulysses-design.png')) {
      expect(entry.sha256).toBe('0c41d8e6105bfda47d57280eb992d450a5ed9dcbdcd820d3a9663b8f9346903d');
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([1408, 649]);
      expect(entry.source_url).toBe('https://arxiv.org/html/2309.14509v2#S3.F2');
      expect(entry.source_image_url).toBe('https://arxiv.org/html/2309.14509v2/figs/image3.png');
      expect(entry.license).toBe('CC BY 4.0');
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
      expect(entry.commit).toBeUndefined();
      expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
        .toContain('http://creativecommons.org/licenses/by/4.0/');
    } else {
      const pin = 'c93890dd4b22356ee18ed9b863e6f2d0909729bc';
      expect(entry.asset.endsWith('hf-ring-attention.gif')).toBe(true);
      expect(entry.sha256).toBe('ac30e82373020c1248fed596c85a86ccc2934c1920e8b9b82b9a8d42d0a055a1');
      expect(entry.commit).toBe(pin);
      expect(entry.source_url).toBe(`https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/${pin}/assets/images/ring-attention.gif`);
      expect(entry.source_image_url).toBe(entry.source_url!.replace('/blob/', '/resolve/'));
      expect(entry.license).toBe('Apache-2.0');
      expect(entry.license_url).toBe(`https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/${pin}/README.md`);
      const bytes = readFileSync(resolve(repositoryRoot, entry.asset));
      expect([bytes.readUInt16LE(6), bytes.readUInt16LE(8)]).toEqual([3000, 1688]);
      expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).toContain('license: apache-2.0');
    }
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    for (const [path, hash] of [[entry.asset, entry.sha256],
      [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/checkpoint-2026-10-23/')) {
    const specs = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/checkpoint-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    const spec = specs.find(row => row.asset === entry.asset);
    expect(spec).toBeTruthy();
    for (const [key, value] of Object.entries(spec!))
      expect((entry as unknown as Record<string, unknown>)[key]).toEqual(value);
    const names: Record<string, [string, string, number, number]> = {
      'pytorch-eager-saved-tensors.png': ['fg3.png', '3cf959c8e9dd01d0cb5dc230849ce570134ac655f38c076378075419ba2c310d', 1466, 1068],
      'pytorch-memory-budget-pareto.png': ['fg12.png', '02c989ad39fc3c3f42f747674945a26ff060079898648a74a622e72fa1b99b3f', 1402, 1032]
    };
    const [name, digest, width, height] = names[entry.asset.split('/').at(-1)!];
    const pin = '563f92c6e87782196b81894c4e671428cd8d34af';
    expect(entry.commit).toBe(pin);
    expect(entry.sha256).toBe(digest);
    expect(entry.source_asset_sha256).toBe(digest);
    expect(entry.source_url).toBe(`https://github.com/pytorch/pytorch.github.io/blob/${pin}/assets/images/activation-checkpointing-techniques/${name}`);
    expect(entry.source_image_url).toBe(`https://raw.githubusercontent.com/pytorch/pytorch.github.io/${pin}/assets/images/activation-checkpointing-techniques/${name}`);
    expect(entry.license).toBe('BSD-3-Clause');
    expect(entry.license_url).toBe(`https://github.com/pytorch/pytorch.github.io/blob/${pin}/LICENSE`);
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.crop_box_px).toBeUndefined();
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([width, height]);
    for (const [path, hash] of [[entry.asset, entry.sha256],
      [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).toContain('Copyright (c) 2018, Facebook Inc');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/ddp-2026-10-23/')) {
    const [spec] = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/ddp-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    for (const [key, value] of Object.entries(spec))
      expect((entry as unknown as Record<string, unknown>)[key]).toEqual(value);
    expect(entry.asset).toBe('00 Учебник/Assets/Figures/curated/ddp-2026-10-23/edls-error-feedback.png');
    expect(entry.commit).toBe('e632aa89ca9e6638d52e1b686095e7442faffbb0');
    expect(entry.source_asset_sha256).toBe('d81b47f8eb116db698a9d9262c7c86ed9dbf4f8bf602fe958a6eaec104c6ef01');
    expect(entry.sha256).toBe('bf4bb0c300d6914b171aca02e39a3753c225f154fbfaed63da07e3aed580c911');
    expect(entry.source_page).toBe(58);
    expect(entry.license).toBe('MIT');
    expect(entry.derivation).toBe('pdf-page-render');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.source_url).toBe(`https://github.com/mryab/efficient-dl-systems/blob/${entry.commit}/week03_data_parallel/lecture.pdf#page=58`);
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.render_page_size_px).toEqual([1920, 1081]);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([1920, 1081]);
    for (const [path, hash] of [[entry.asset, entry.sha256],
      [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).toContain('Copyright (c) 2021 course authors');
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/distributed-overview-2026-10-23/')) {
    const specs = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/distributed-overview-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    const spec = specs.find(row => row.asset === entry.asset);
    expect(spec).toBeTruthy();
    for (const [key, value] of Object.entries(spec!))
      expect((entry as unknown as Record<string, unknown>)[key]).toEqual(value);
    const names: Record<string, [string, string]> = {
      'hf-ddp-sequential.svg': ['dp_overlap1.svg', 'd622641a5e1ffa561888558cdb1b310ff5081562b5edeaf0abc153a1bedb7b66'],
      'hf-ddp-overlap.svg': ['dp_overlap2.svg', '43e29a970b78972837688fe9dd5f8c2fd581f39eb9e8ebf1f59488b6b465cf23']
    };
    const [source, digest] = names[entry.asset.split('/').at(-1)!];
    const pin = 'c93890dd4b22356ee18ed9b863e6f2d0909729bc';
    expect(entry.commit).toBe(pin);
    expect(entry.sha256).toBe(digest);
    expect(entry.source_asset_sha256).toBe(digest);
    expect(entry.source_url).toBe(`https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/${pin}/assets/images/${source}`);
    expect(entry.source_image_url).toBe(entry.source_url!.replace('/blob/', '/resolve/'));
    expect(entry.source_asset).toBe('05 Источники/Originals/distributed-overview-2026-10-23/' + source);
    expect(entry.license).toBe('Apache-2.0');
    expect(entry.license_url).toBe(`https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/${pin}/README.md`);
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    expect(entry.derivation).toBe('source-svg-copy');
    expect(entry.metadata_status).toBe('verified');
    for (const [path, hash] of [[entry.asset, entry.sha256],
      [entry.source_asset, entry.source_asset_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256],
      [entry.license_evidence, entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
    expect(readFileSync(resolve(repositoryRoot, entry.asset))).toEqual(
      readFileSync(resolve(repositoryRoot, entry.source_asset!)));
    expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8'))
      .toContain('license: apache-2.0');
    expect(entry.crop_box_px).toBeUndefined();
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/ppo-2026-10-24/')) {
    const pin = '3f573487c58548c16a548cd5fb8979a0189384c7';
    const digest = '5871236be3cbdc8e2c3732304e6384e59b46dbeaa320f349751570d153172bd0';
    expect(entry.asset).toBe('00 Учебник/Assets/Figures/curated/ppo-2026-10-24/ppo-framework.png');
    expect(entry.commit).toBe(pin);
    expect(entry.source_url).toBe(`https://github.com/natolambert/rlhf-book/blob/${pin}/book/images/ppo_tikz.png`);
    expect(entry.source_image_url).toBe(`https://raw.githubusercontent.com/natolambert/rlhf-book/${pin}/book/images/ppo_tikz.png`);
    expect(entry.license).toBe('MIT');
    expect(entry.license_url).toBe(`https://github.com/natolambert/rlhf-book/blob/${pin}/LICENSE-CODE`);
    expect(entry.license_notice).toBe('site/public/licenses/rlhf-book-MIT.txt');
    expect(entry.derivation).toBe('source-raster-copy');
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    expect(entry.metadata_status).toBe('verified');
    expect(entry.sha256).toBe(digest);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(digest);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([2158, 707]);
    expect(entry.crop_box_px).toBeUndefined();
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/harvard-robustness-2026-11-09/fgsm-geometry.svg') {
    const source = 'c6861733e4dd4202ae4a396c96e3abe87ae103c7';
    const release = '7367c67c2afd6cbb42b9b2ac98cc673d2f5bba37';
    const hash = 'a27bd92b8a2649abed8e2913ace4d3f8bcb61ac0f5195aa1f1966444604c633b';
    const repo = 'https://github.com/harvard-edge/cs249r_book/blob/';
    const raw = 'https://raw.githubusercontent.com/harvard-edge/cs249r_book/';
    expect(entry).toMatchObject({
      commit: source, source_render_commit: release, sha256: hash, source_asset_sha256: hash,
      source_url: repo + source + '/books/vol2/14_robust_ai/14_robust_ai.qmd#L1232',
      source_image_url: raw + source + '/books/vol2/14_robust_ai/images/svg/gradient-attack.svg',
      source_published_image_url: raw + release + '/vol2/14_robust_ai/images/svg/gradient-attack.svg',
      license_evidence: repo + source + '/books/vol2/tex/copyright.tex',
      license: 'CC BY-NC-ND 4.0', license_url: 'https://creativecommons.org/licenses/by-nc-nd/4.0/',
      derivation: 'source-svg-copy', metadata_status: 'verified', provenance_confirmation: 'source-license-verified',
    });
    expect(entry.crop_box_px).toBeUndefined();
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(hash);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/scaling-lifetime-2026-11-14/sardana24a-figure5.png') {
    // Figure 5 is the RGB image object in the paper, not a screenshot of a lecture.
    expect(entry).toMatchObject({
      source_url: 'https://proceedings.mlr.press/v235/sardana24a.html',
      source_image_url: 'https://raw.githubusercontent.com/mlresearch/v235/main/assets/sardana24a/sardana24a.pdf#page=6',
      source_asset: '05 Источники/Originals/llama-scaling-2026-11-09/sardana24a.pdf',
      source_asset_sha256: 'cd66b62454f51dcee723ba98746f7ae37e5ff0c7dce2441b9fe3d6decd188952',
      source_page: 6, source_pdf_object: 32, dimensions_px: [1710, 1314],
      sha256: '2eeafe50cf15461f81656df20cfbe90302b4ec97bdb09f7a1b7834892038094c',
      license: 'CC BY 4.0', license_url: 'https://creativecommons.org/licenses/by/4.0/',
      license_evidence_url: 'https://proceedings.mlr.press/pmlr-license-agreement.html',
      derivation: 'pdf-embedded-raster-extraction', rights_status: 'source-license-verified',
      provenance_confirmation: 'source-license-verified', metadata_status: 'verified',
      used_in: ['00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md'],
    });
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.render_dpi).toBeUndefined();
    const source = resolve(repositoryRoot, entry.source_asset!);
    if (existsSync(source)) expect(sha256(source)).toBe(entry.source_asset_sha256);
    const target = resolve(repositoryRoot, entry.asset);
    const png = readFileSync(target);
    expect(png.length).toBe(entry.bytes);
    expect(sha256(target)).toBe(entry.sha256);
    expect(validateRenderedPngBytes(png)).toEqual(entry.dimensions_px);
    const chapter = readFileSync(resolve(repositoryRoot, entry.used_in![0]!), 'utf8');
    for (const reference of [entry.asset + '|', entry.source_image_url, entry.license_url])
      expect(chapter).toContain(reference);
    return;
  }
  const aclPaper = entry.source_url?.match(/^https:\/\/aclanthology\.org\/\d{4}\.[\w-]+\.\d+\/$/);
  const htmlLocation = aclPaper && entry.derivation === 'source-svg-copy' ? entry.figure_url : entry.source_url;
  const htmlPaper = htmlLocation?.match(/^https:\/\/arxiv\.org\/html\/(\d{4}\.\d{4,5}v[1-9]\d*)(?:#([\w.:-]+))?$/);
  if (!entry.source_asset && htmlPaper && entry.provenance_confirmation === 'source-license-verified') {
    // The canonical copy may be the downloaded HTML figure, rather than
    // a second local copy. Bind the figure and evidence to this paper version.
    const paper = htmlPaper[1]!;
    const article = 'https://arxiv.org/html/' + paper;
    const image = entry.source_image_url ?? '';
    const target = resolve(repositoryRoot, entry.asset);
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    const cc = entry.license?.match(/^CC (BY(?:-NC)?(?:-SA|-ND)?) 4\.0$/);
    expect(cc, entry.asset).toBeTruthy();
    expect(entry.license_url).toBe('https://creativecommons.org/licenses/' + cc![1]!.toLowerCase() + '/4.0/');
    expect(entry.metadata_status).toBe('verified');
    const evidence = entry.license_evidence ?? entry.license_evidence_url ?? article;
    if (evidence.startsWith('https://')) {
      expect(aclPaper
        ? [article, 'https://arxiv.org/abs/' + paper]
        : [article]).toContain(evidence);
    } else {
      expect(entry.license_evidence_sha256).toMatch(/^[a-f0-9]{64}$/);
      if (entry.source_page_snapshot) {
        expect(evidence).toBe(entry.source_page_snapshot);
        expect(sha256(resolve(repositoryRoot, evidence))).toBe(entry.license_evidence_sha256);
      } else {
        // An independently archived HTML license page need not duplicate the
        // source-page field. Public clones may not have the local archive.
        expect(evidence).toMatch(/^05 Источники\/Originals\/[\w-]+\/[\w.-]+\.html?$/);
        const snapshot = resolve(repositoryRoot, evidence);
        if (existsSync(snapshot)) {
          expect(sha256(snapshot)).toBe(entry.license_evidence_sha256);
          const html = readFileSync(snapshot, 'utf8');
          expect(html).toContain('License: ' + entry.license);
          expect(html).toContain('id="' + htmlPaper[2] + '"');
          expect(html).toContain(image.slice('https://arxiv.org/html/'.length));
        }
      }
    }
    if (['source-svg-copy', 'source-raster-copy'].includes(entry.derivation ?? '')) {
      const file = image.slice(article.length + 1);
      const extension = entry.derivation === 'source-svg-copy' ? 'svg' : '(png|jpg|jpeg|webp)';
      const stem = extension === 'svg' ? '[a-zA-Z0-9_-]+(?:\\.[a-zA-Z0-9_-]+)*' : '[a-zA-Z0-9_-]+';
      expect(image.startsWith(article + '/')).toBe(true);
      expect(file).toMatch(new RegExp(`^(?:[a-zA-Z0-9_-]+/)*${stem}\\.${extension}$`));
      expect(entry.asset).toMatch(new RegExp(`\\.${extension}$`));
      if (entry.source_asset_sha256) expect(entry.sha256).toBe(entry.source_asset_sha256);
      expect(entry.crop_box_px).toBeUndefined();
    } else {
      expect(entry.source_asset_sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(entry.asset).toMatch(/\.png$/);
      expect(pngDimensions(target)).toEqual(entry.dimensions_px);
      if (entry.derivation === 'source-pdf-render') {
        expect(image).toBe('https://arxiv.org/src/' + paper);
        expect(entry.source_archive_member).toMatch(/^(?:[a-zA-Z0-9_-]+\/)*[a-zA-Z0-9_-]+\.pdf$/);
        expect(entry.render_dpi).toBeGreaterThan(0);
        expect(entry.crop_box_px).toBeUndefined();
      } else {
        expect(entry.derivation).toBe('pdf-page-render-crop');
        expect(entry.source_page).toBeGreaterThan(0);
        expect(image).toBe('https://arxiv.org/pdf/' + paper + '#page=' + entry.source_page);
      }
    }
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(target)).toBe(entry.sha256);
    expectDecodableImage(target);
    if (entry.derivation === 'source-svg-copy' && entry.bytes !== undefined && htmlPaper[2]) {
      const svg = readFileSync(target, 'utf8');
      expect(Buffer.byteLength(svg)).toBe(entry.bytes);
      expect(entry.source_asset_sha256).toBe(entry.sha256);
      expect(svg).not.toMatch(/<\s*(?:script|foreignObject)[\s>]|[\s]on[a-z]+\s*=/i);
      if (entry.source_viewbox) {
        const viewBox = svg.match(/<svg\b[^>]*\bviewBox="([^"]+)"/s)?.[1]?.split(/[\s,]+/).map(Number);
        expect(viewBox).toEqual(entry.source_viewbox);
      }
      for (const owner of entry.used_in ?? []) {
        const chapter = readFileSync(resolve(repositoryRoot, owner), 'utf8');
        for (const reference of [entry.asset + '|', htmlLocation, image, entry.license_url])
          expect(chapter).toContain(reference);
      }
    }
    return;
  }
  if (/\/curated\/astra\/original-(?:env|sft-pipeline)\.png$/.test(entry.asset)) {
    const commit = 'bdf6a46a130118e039aadcaeb46724d278383589';
    const name = entry.asset.split('/').at(-1)!.replace('original-', '');
    const repo = 'https://github.com/LianjiaTech/astra/blob/' + commit;
    const notice = 'site/public/licenses/astra-bdf6a46-Apache-2.0.txt';
    expect(entry).toMatchObject({
      commit, source_url: repo + '/assets/' + name,
      source_image_url: 'https://raw.githubusercontent.com/LianjiaTech/astra/' + commit + '/assets/' + name,
      license: 'Apache-2.0', license_url: repo + '/licenses/LICENSE',
      license_evidence: notice, public_license_copy: notice,
      metadata_status: 'verified', provenance_confirmation: 'source-license-verified',
      derivation: 'source-raster-copy',
    });
    expect(sha256(resolve(repositoryRoot, notice))).toBe(entry.license_evidence_sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual(entry.dimensions_px);
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/atlas-courses-official/qwen3-post-training-pipeline.png') {
    // Preserve the historical crop without treating the source paper as reuse permission.
    expect(entry).toMatchObject({
      source_url: 'https://arxiv.org/pdf/2505.09388',
      source_asset: 'raw/papers/qwen3/paper.pdf',
      provenance_confirmation: 'original-paper-figure',
      publication_status: 'retired', used_in: [],
      rights_status: 'paper-figure-license-unverified', metadata_status: 'needs-confirmation',
      derivation: 'pdf-page-render-crop',
    });
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/moba/moba-running-example.png') {
    const commit = 'b5d58363311d3ca946f1ec444182727c15e338b5';
    const notice = 'site/public/licenses/moonshot-moba-b5d58363-MIT.txt';
    // Figure explanation is in the paper; the image and its license are in the author repository.
    expect(entry).toMatchObject({
      commit, source_url: 'https://arxiv.org/html/2502.13189v1#S2.F1',
      source_image_url: 'https://raw.githubusercontent.com/MoonshotAI/MoBA/' + commit + '/figures/running_example.png',
      license: 'MIT', license_url: 'https://github.com/MoonshotAI/MoBA/blob/' + commit + '/LICENSE',
      license_evidence: notice, metadata_status: 'verified',
      provenance_confirmation: 'pinned-upstream-repository', derivation: 'source-image-copy',
    });
    expect(sha256(resolve(repositoryRoot, notice))).toBe(entry.license_evidence_sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual(entry.dimensions_px);
    return;
  }
  if (entry.provenance_confirmation === 'source-license-verified'
    && (!entry.source_asset || ((entry.source_asset.startsWith('05 Источники/Originals/')
      || entry.source_asset === entry.asset)
      && entry.source_url?.startsWith('https://arxiv.org/abs/')
      && ['source-raster-copy', 'source-svg-copy'].includes(entry.derivation ?? '')))) {
    // arXiv HTML exposes unchanged original figures under a versioned paper URL.
    // Retain both source and output hashes; never accept an unversioned latest URL.
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.license, 'license');
    expect(entry.license_url).toMatch(/^https:\/\//);
    expect(entry.metadata_status).toBe('verified');
    expect(['source-raster-copy', 'source-svg-copy']).toContain(entry.derivation);
    const extension = entry.derivation === 'source-svg-copy' ? 'svg' : '(png|jpg|jpeg|webp)';
    expect(entry.asset).toMatch(new RegExp(`\\.${extension}$`));
    const paper = entry.source_url?.match(/^https:\/\/arxiv\.org\/abs\/(\d{4}\.\d{4,5}v[1-9]\d*)$/);
    expect(paper).toBeTruthy();
    expect(entry.source_image_url).toMatch(new RegExp(
      `^https://arxiv\\.org/html/${paper?.[1].replace('.', '\\.')}/(?:[a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+\\.${extension}$`
    ));
    expect(entry.source_asset_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(entry.sha256).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    requireNonEmptyString(entry.modifications, 'modifications');
    if (entry.source_asset) {
      expect(entry.source_asset.split('/').some(part => part === '.' || part === '..')).toBe(false);
      expect(sha256(resolve(repositoryRoot, entry.source_asset))).toBe(entry.source_asset_sha256);
      expect(readFileSync(resolve(repositoryRoot, entry.asset))).toEqual(readFileSync(resolve(repositoryRoot, entry.source_asset)));
      for (const [path, hash] of [[entry.source_page_snapshot, entry.source_page_sha256],
        [entry.license_evidence, entry.license_evidence_sha256]]) {
        expect(hash).toMatch(/^[a-f0-9]{64}$/);
        expect(sha256(resolve(repositoryRoot, path!))).toBe(hash);
      }
      expect(entry.crop_box_px).toBeUndefined();
      expectDecodableImage(resolve(repositoryRoot, entry.asset));
    }
    return;
  }
  const decoderSlide = decoderBlockFigures.find((s: CuratedAsset) => s.asset === entry.asset);
  if (decoderSlide) {
    for (const key of ['sha256', 'source_asset', 'source_asset_sha256', 'source_page',
      'render_dpi', 'render_page_size_px', 'commit', 'derivation', 'rights_status',
      'metadata_status', 'license'] as const) expect(entry[key]).toEqual(decoderSlide[key]);
    expect(entry.source_url).toBe('https://github.com/stanford-cs336/lectures/blob/'
      + decoderSlide.commit + '/lecture_03.pdf#page=' + decoderSlide.source_page);
    expect(entry.crop_box_px).toBeUndefined();
    expect(sha256(resolve(repositoryRoot, entry.source_asset!))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([3600, 2027]);
    validateRenderedPngBytes(readFileSync(resolve(repositoryRoot, entry.asset)));
    return;
  }

  if (entry.asset === '00 Учебник/Assets/Figures/curated/profiling-2026-11-04/py-spy-flamegraph.svg') {
    const pin='7a7ac96e96e0f9b4ba3eee80de3f4d255698170b';
    const expected={
      author:'Ben Frederickson',license:'MIT',commit:pin,derivation:'source-svg-copy',
      source_url:`https://github.com/benfred/py-spy/blob/${pin}/README.md#record`,
      source_image_url:`https://raw.githubusercontent.com/benfred/py-spy/${pin}/images/flamegraph.svg`,
      source_asset:'publishing/fixtures/profiling-originals/py-spy-flamegraph.svg',
      source_asset_sha256:'592970e72ed1c0f0f6e026a63187ffef3e91eab45d0c5457ca6649388aa3de9a',
      sha256:'592970e72ed1c0f0f6e026a63187ffef3e91eab45d0c5457ca6649388aa3de9a',
      source_page_snapshot:'publishing/fixtures/profiling-originals/py-spy-README.md',
      source_page_sha256:'b94fab51768821fca5fdea943a80d799fd046d9bfa3cb21623cffd900f57567f',
      license_url:`https://github.com/benfred/py-spy/blob/${pin}/LICENSE`,
      license_evidence:'site/public/licenses/py-spy-7a7ac96e-MIT.txt',
      license_evidence_sha256:'80bbb8731db59cd835f59fcd06f127953804bb511c8487fd265d96c7c702cd00',
      metadata_status:'verified',provenance_confirmation:'pinned-upstream-repository'
    };
    for(const [key,value]of Object.entries(expected))
      expect((entry as unknown as Record<string,unknown>)[key]).toBe(value);
    for(const [file,digest]of [[entry.asset,entry.sha256],[entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],[entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,file!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(
      readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.source_viewbox).toEqual([0,0,1200,182]);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }
  if (entry.asset === '00 Учебник/Assets/Figures/curated/dali-2026-11-04/two_readers.svg') {
    const pin = '5b33fee1968ba069a9ed04965fd2af5f0025de7e';
    const packet = 'publishing/fixtures/dali-originals/';
    const expected = {
      source_url: `https://github.com/NVIDIA/DALI/blob/${pin}/docs/pipeline.rst`,
      source_image_url: `https://raw.githubusercontent.com/NVIDIA/DALI/${pin}/docs/images/two_readers.svg`,
      commit: pin, derivation: 'source-svg-copy', license: 'Apache-2.0',
      author: 'NVIDIA DALI contributors',
      source_asset: packet + 'two_readers.svg',
      source_asset_sha256: '3a04306186cc2d4c1800a1c9e9b2bd7051646ec35c4054ace1cac1225da17214',
      sha256: '3a04306186cc2d4c1800a1c9e9b2bd7051646ec35c4054ace1cac1225da17214',
      source_page_snapshot: packet + 'pipeline.rst',
      source_page_sha256: '3ca2cc06792e49f6f9280bf9f745a85938c603d53bb01b44743d1414eff04be3',
      license_url: `https://github.com/NVIDIA/DALI/blob/${pin}/LICENSE`,
      license_evidence: 'site/public/licenses/nvidia-dali-5b33fee1-Apache-2.0.txt',
      license_evidence_sha256: 'bea1b4266f1499fa4bbb16d6e7577e6b5b53d1ef010835a9d460876022cc8722',
      metadata_status: 'verified', provenance_confirmation: 'pinned-upstream-repository'
    };
    for (const [key,value] of Object.entries(expected))
      expect((entry as unknown as Record<string,unknown>)[key]).toBe(value);
    for (const [file,digest] of [[entry.asset,entry.sha256],
      [entry.source_asset,entry.source_asset_sha256],
      [entry.source_page_snapshot,entry.source_page_sha256],
      [entry.license_evidence,entry.license_evidence_sha256]])
      expect(sha256(resolve(repositoryRoot,file!))).toBe(digest);
    expect(readFileSync(resolve(repositoryRoot,entry.asset))).toEqual(
      readFileSync(resolve(repositoryRoot,entry.source_asset!)));
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.source_viewbox).toEqual([0,0,143.99135,87.982686]);
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }

  // A raster copied from a pinned repository can retain its original tracked
  // binary as well as the published copy, without pretending to be a PDF render.
  if (entry.provenance_confirmation === 'pinned-upstream-repository'
    && entry.derivation === 'source-raster-copy'
    && entry.source_asset?.startsWith('05 Источники/Originals/')) {
    const source = entry.source_url?.match(
      /^https:\/\/github\.com\/([^/]+\/[^/]+)\/blob\/([0-9a-f]{40})\/(.+\.(?:png|jpe?g|webp))$/
    );
    expect(source).toBeTruthy();
    expect(source?.[2]).toBe(entry.commit);
    expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/'
      + source?.[1] + '/' + source?.[2] + '/' + source?.[3]);
    expect(entry.metadata_status).toBe('verified');
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.license, 'license');
    requireNonEmptyString(entry.modifications, 'modifications');
    expect(entry.license_url).toMatch(/^https:\/\//);
    for (const [file, digest] of [[entry.asset, entry.sha256],
      [entry.source_asset, entry.source_asset_sha256],
      [entry.license_evidence, entry.license_evidence_sha256],
      [entry.source_page_snapshot, entry.source_page_sha256]]) {
      expect(digest).toMatch(/^[a-f0-9]{64}$/);
      expect(sha256(resolve(repositoryRoot, file!))).toBe(digest);
    }
    expect(entry.sha256).toBe(entry.source_asset_sha256);
    expect(readFileSync(resolve(repositoryRoot, entry.asset))).toEqual(
      readFileSync(resolve(repositoryRoot, entry.source_asset)));
    expect(entry.crop_box_px).toBeUndefined();
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.derivation === 'source-svg-style-bundle') {
    const pin = 'eb56ba22fd2b3c1fa4c962a4a575d4ba12d1fcbc';
    const originals: Record<string, { path: string; svg: string; css: string }> = {
      '00 Учебник/Assets/Figures/curated/precision-review-2026-09/te-current-delayed-scaling.svg': {
        path: 'docs/features/low_precision_training/fp8_delayed_scaling/img/scaling_comparison.svg',
        svg: '05 Источники/Originals/precision-2026-10-23/scaling_comparison.svg',
        css: '05 Источники/Originals/precision-2026-10-23/diagram-colors.css'
      },
      '00 Учебник/Assets/Figures/curated/precision-2026-11/te-mxfp8-row-column.svg': {
        path: 'docs/features/low_precision_training/mxfp8/img/mxfp8_row_col.svg',
        svg: 'publishing/fixtures/precision-originals/mxfp8_row_col.svg',
        css: 'publishing/fixtures/precision-originals/diagram-colors.css'
      }
    };
    expect(originals[entry.asset], 'Only the two reviewed complete TE diagrams are allowed').toBeDefined();
    const original = originals[entry.asset]!;
    const sourcePath = original.path;
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    expect(entry.commit).toBe(pin);
    expect(entry.license).toBe('Apache-2.0');
    expect(entry.license_url).toBe(`https://github.com/NVIDIA/TransformerEngine/blob/${pin}/LICENSE`);
    expect(entry.source_url).toBe(`https://github.com/NVIDIA/TransformerEngine/blob/${pin}/${sourcePath}`);
    expect(entry.source_image_url).toBe(`https://raw.githubusercontent.com/NVIDIA/TransformerEngine/${pin}/${sourcePath}`);
    expect(entry.source_asset).toBe(original.svg);
    expect(entry.source_stylesheet_asset).toBe(original.css);
    expect(entry.source_stylesheet_url).toBe(`https://raw.githubusercontent.com/NVIDIA/TransformerEngine/${pin}/docs/_static/css/diagram-colors.css`);
    expect(entry.metadata_status).toBe('verified');
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.modifications, 'modifications');
    for (const [file, hash] of [[entry.source_asset, entry.source_asset_sha256],
      [entry.source_stylesheet_asset, entry.source_stylesheet_sha256], [entry.asset, entry.sha256]]) {
      expect(hash).toMatch(/^[a-f0-9]{64}$/);
      expect(sha256(resolve(repositoryRoot, file!))).toBe(hash);
    }
    const source = readFileSync(resolve(repositoryRoot, entry.source_asset!), 'utf8');
    const css = readFileSync(resolve(repositoryRoot, entry.source_stylesheet_asset!), 'utf8');
    const notice = '  <!-- Bookvar: unchanged upstream stylesheet embedded for standalone rendering; no geometry or labels changed. -->\n';
    const bundle = notice + '  <style><![CDATA[\n' + css + ']]></style>\n';
    expect(readFileSync(resolve(repositoryRoot, entry.asset), 'utf8')).toBe(
      source.replace('  <defs>\n', bundle + '  <defs>\n'));
    return;
  }
  if (!entry.source_asset) {
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    if (entry.source_url?.startsWith('https://huggingface.co/')) {
      const sourceMatch = entry.source_url.match(
        /^https:\/\/huggingface\.co\/(?:spaces|datasets)\/([\w-]+\/[\w.-]+)\/blob\/([0-9a-f]{40})\/((?:[\w-]+\/)*[\w-]+\.(svg|png))$/
      );
      expect(sourceMatch, `Curated Hugging Face source must be a pinned Space or Dataset SVG or PNG: ${entry.asset}`).toBeTruthy();
      expect(entry.commit).toMatch(/^[0-9a-f]{40}$/);
      expect(sourceMatch?.[2]).toBe(entry.commit);
      expect(entry.source_image_url).toBe(entry.source_url.replace('/blob/', '/resolve/'));
      requireNonEmptyString(entry.author, 'author');
      requireNonEmptyString(entry.license, 'license');
      requireNonEmptyString(entry.modifications, 'modifications');
      expect(entry.license_url).toMatch(/^https:\/\//);
      expect(entry.metadata_status).toBe('verified');
      expect(entry.derivation).toBe(sourceMatch?.[4] === 'svg' ? 'source-svg-copy' : 'source-raster-copy');
      expect(entry.asset).toMatch(/^00 Учебник\/Assets\/Figures\/curated\/(?:[\w-]+\/)+[\w-]+\.(svg|png)$/);
      expect(extname(entry.asset)).toBe(`.${sourceMatch?.[4]}`);
      expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
      expectDecodableImage(resolve(repositoryRoot, entry.asset));
      return;
    }
    const sourceMatch = entry.source_url?.match(
      /^https:\/\/github\.com\/[^/]+\/[^/]+\/blob\/([0-9a-f]{40})\/.+/
    );
    expect(sourceMatch, `Curated upstream source must be a pinned GitHub blob: ${entry.asset}`).toBeTruthy();
    expect(entry.commit).toMatch(/^[0-9a-f]{40}$/);
    expect(sourceMatch?.[1]).toBe(entry.commit);
    const archivedScaling = {
      'deepseek-hyperparameter-fits.png': [21, 'bcd69487081eb1c26301854526a8982da361088371ff57828fed1d4f23bad23a',
        [1230, 475], 'https://arxiv.org/html/2401.02954v1#S3.F3'],
      'optimizer-extrapolation-failure.png': [41, '2e831ffa5d05d077fd951acdac3b8989e71e30e4850991bd29de2691cd5dc572',
        [890, 450], 'https://openathena.ai/blog/delphi/#delphi-attempt-1-how-can-scaling-laws-go-wrong'],
    } as const;
    const scalingPath = '00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/';
    const scalingName = entry.asset.slice(scalingPath.length);
    if (entry.asset.startsWith(scalingPath) && Object.hasOwn(archivedScaling, scalingName)) {
      const [page, hash, size, figure] = archivedScaling[scalingName as keyof typeof archivedScaling];
      expect(entry).toMatchObject({
        source_url: `https://github.com/stanford-cs336/lectures/blob/${cs336Commit}/lecture_11.pdf#page=${page}`,
        source_parent_sha256: 'b2148e683b4020f3fa95f5c95f37be55ebf6b2f57382f4c148517e84bf1df890',
        figure_url: figure, sha256: hash, derivation: 'pdf-page-crop', used_in: [],
        publication_status: 'retired', metadata_status: 'needs-confirmation',
        rights_status: 'existing-local-reformat; third-party-rights-unverified',
      });
      requireNonEmptyString(entry.retirement_reason, 'retirement_reason');
      const path = resolve(repositoryRoot, entry.asset);
      expect(sha256(path)).toBe(hash);
      expect(pngDimensions(path)).toEqual(size);
    } else {
      expect(entry.metadata_status).toBe('verified');
    }
    return;
  }

  if (entry.derivation === 'source-svg-crop') {
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    expect(entry.source_asset).toMatch(/^00 Учебник\/Assets\/Figures\/curated\/(?:[\w-]+\/)+[\w-]+\.svg$/);
    expect(entry.asset).toMatch(/\.svg$/);
    const pinned = entry.source_url?.match(/^https:\/\/github\.com\/[^/]+\/[^/]+\/blob\/([a-f0-9]{40})\/.+\.svg$/);
    expect(pinned).toBeTruthy();
    expect(pinned?.[1]).toBe(entry.commit);
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.license, 'license');
    requireNonEmptyString(entry.modifications, 'viewport crop description');
    expect(entry.license_url).toMatch(/^https:\/\//);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.source_asset_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    const source = resolve(repositoryRoot, entry.source_asset!);
    const target = resolve(repositoryRoot, entry.asset);
    expect(sha256(source)).toBe(entry.source_asset_sha256);
    expect(sha256(target)).toBe(entry.sha256);
    expectViewportOnlySvgCrop(readFileSync(source, 'utf8'), readFileSync(target, 'utf8'));
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
  if (entry.provenance_confirmation === 'source-license-verified'
    && entry.derivation === 'source-raster-crop') {
    expect(entry.source_asset).toMatch(/^00 Учебник\/Assets\/Figures\/curated\/(?:[^/]+\/)*[^/]+\.png$/i);
    expect(entry.source_asset.split('/').some(segment => segment === '.' || segment === '..')).toBe(false);
    expect(entry.asset).toMatch(/\.png$/i);
    requireNonEmptyString(entry.modifications, 'crop description');
    expect(entry.source_asset_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.source_asset))).toBe(entry.source_asset_sha256);
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(Array.isArray(entry.crop_box_px)).toBe(true);
    const crop = entry.crop_box_px as number[];
    expect(crop).toHaveLength(4);
    expect(crop.every(Number.isInteger)).toBe(true);
    const [left, top, right, bottom] = crop;
    const [width, height] = pngDimensions(resolve(repositoryRoot, entry.source_asset));
    expect(left).toBeGreaterThanOrEqual(0);
    expect(top).toBeGreaterThanOrEqual(0);
    expect(right).toBeGreaterThan(left);
    expect(bottom).toBeGreaterThan(top);
    expect(right).toBeLessThanOrEqual(width);
    expect(bottom).toBeLessThanOrEqual(height);
    expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([right - left, bottom - top]);
    return;
  }
  if (entry.provenance_confirmation === 'source-license-verified'
    && entry.derivation === 'lossless-format-conversion'
    && entry.source_asset.startsWith('00 Учебник/Assets/Figures/curated/')) {
    expect(entry.source_asset).toMatch(/^00 Учебник\/Assets\/Figures\/curated\/(?:[^/]+\/)*[^/]+\.jpe?g$/i);
    expect(entry.source_asset.split('/').some(segment => segment === '.' || segment === '..')).toBe(false);
    expect(entry.asset).toMatch(/\.png$/i);
    requireNonEmptyString(entry.modifications, 'conversion description');
    expect(entry.source_asset_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.source_asset))).toBe(entry.source_asset_sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expectDecodableImage(resolve(repositoryRoot, entry.source_asset));
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/formal-zeta-2026-10-26/')) {
    const specs = JSON.parse(readFileSync(resolve(repositoryRoot,
      '00 Учебник/Assets/Figures/curated/formal-zeta-2026-10-26/assets.json'), 'utf8')) as CuratedAsset[];
    const spec = specs.find(row => row.asset === entry.asset);
    expect(spec).toBeTruthy();
    for (const [key, value] of Object.entries(spec!))
      expect((entry as unknown as Record<string, unknown>)[key]).toEqual(value);
    expect(entry.derivation).toBe('pdf-embedded-raster-copy');
    expect(entry.license).toBe('User permission record (not a license)');
    expect(entry.rights_status).toBe('permission-recorded');
    expect(entry.source_asset).toBe('05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/Lectures/meeting-08-slides.pdf');
    expect(entry.source_asset_sha256).toBe('94bd68c2fdb534d57459a8b96f2a80ada96fd14a58b36626016d4a9f601866ab');
    expect([99, 100]).toContain(entry.source_page);
    expect(entry.source_pdf_object).toBe(entry.source_page === 99 ? 638 : 645);
    expect(entry.source_url).toBe(`https://rdi.berkeley.edu/adv-llm-agents/slides/alphaproof.pdf#page=${entry.source_page}`);
    expect(entry.crop_box_px).toBeUndefined();
    expect(entry.render_dpi).toBeUndefined();
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset))).toBe(entry.source_asset_sha256);
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
  }
  // Some legacy rights records have now acquired tracked, byte-verified raster
  // parents. Check that derivation without upgrading the permission assertion
  // to a named license, or misclassifying a PNG as a rendered lecture PDF.
  if (entry.provenance_confirmation === 'user-confirmed-open-materials'
    && entry.derivation === 'source-raster-copy'
    && entry.source_asset_sha256 && entry.source_image_url) {
    expect(entry.source_asset).toMatch(
      /^(?:00 Учебник\/Assets\/Figures\/curated\/(?:[^/]+\/)*[^/]+|05 Источники\/Courses\/[^/]+\/Lectures\/repository\/images\/[^/]+)\.png$/);
    expect(entry.source_asset.split('/').some(part => part === '.' || part === '..')).toBe(false);
    requireNonEmptyString(entry.author, 'author');
    requireNonEmptyString(entry.license, 'legacy permission record');
    requireNonEmptyString(entry.modifications, 'copy description');
    expect(entry.source_url).toMatch(/^https:\/\//);
    expect(entry.source_image_url).toMatch(/^https:\/\//);
    expect(entry.metadata_status).toBe('verified');
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(entry.source_asset_sha256).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    expect(sha256(resolve(repositoryRoot, entry.source_asset))).toBe(entry.source_asset_sha256);
    expect(entry.crop_box_px).toBeUndefined();
    expectDecodableImage(resolve(repositoryRoot, entry.asset));
    return;
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

  // Public TinyBERT fixtures preserve the two author rasters independently
  // of their curated copies. Figure rights come from ACL, not Apache code.
  if (entry.derivation === 'source-raster-copy'
    && entry.source_asset.startsWith('publishing/fixtures/tinybert-0598f02d/')) {
    const filename=entry.source_asset.split('/').at(-1)!;
    const known: Record<string,{sha:string; dimensions:number[]; url:string}> = {
      'tinybert-overview.png': {
        sha:'3144c0d6af96614ffded2e4097b9d0b7af2743115598fe0d580c05347a00038b',
        dimensions:[1229,320],
        url:'https://raw.githubusercontent.com/huawei-noah/Pretrained-Language-Model/0598f02d7fc4eaa7b4cbc1e9d7ab18bc875c24f9/TinyBERT/tinybert_overview.png',
      },
      'tinybert-layer-detail.png': {
        sha:'c3022bd5d1fad3c9d79cea56c4f99e4aa8f42c5d7a44d03b7e14c6eb1e932f33',
        dimensions:[745,820],
        url:'https://arxiv.org/html/1909.10351v5/transformer_layer_detail.png',
      },
    };
    const expected=known[filename];
    expect(expected,filename).toBeDefined();
    expect(entry.asset).toBe('00 Учебник/Assets/Figures/curated/distillation-tinybert-2026-11-05/'+filename);
    expect(entry.source_image_url).toBe(expected.url);
    expect(entry.source_asset_sha256).toBe(expected.sha);
    expect(entry.sha256).toBe(expected.sha);
    expect(entry.dimensions_px).toEqual(expected.dimensions);
    expect(entry.license).toBe('CC BY 4.0');
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.license_evidence).toBe('publishing/fixtures/tinybert-0598f02d/acl-copyright.html');
    expect(entry.license_evidence_sha256).toBe('d27a4182ffd579b137abf42dab46c7cb836610a616196395dd66b60284b2c237');
    expect(sha256(resolve(repositoryRoot,entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    expect(sha256(resolve(repositoryRoot,entry.source_asset))).toBe(expected.sha);
    expect(sha256(resolve(repositoryRoot,entry.asset))).toBe(expected.sha);
    expect(readFileSync(resolve(repositoryRoot,entry.source_asset)).equals(
      readFileSync(resolve(repositoryRoot,entry.asset)))).toBe(true);
    expect(entry.crop_box_px).toBeUndefined();
    expectDecodableImage(resolve(repositoryRoot,entry.asset));
    return;
  }

  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/normalization-originals-2026-11-09/')) {
    const original = '05 Источники/Originals/normalization-originals-2026-11-09/';
    const name = entry.asset.split('/').at(-1)!;
    expect(entry.provenance_confirmation).toBe('source-license-verified');
    expect(entry.rights_status).toBe('source-license-verified');
    expect(entry.metadata_status).toBe('verified');
    const path = resolve(repositoryRoot, entry.asset);
    if (name === 'fleuret-normalization-axes.png') {
      expect(entry.source_asset).toBe(original + 'lbdl-v1.3.pdf');
      expect(entry.source_asset_sha256).toBe('a5dcbff3eb090dc55bd05d969f610c97a63983d6c07bd736c06d24132019e5e6');
      expect(entry.source_image_url).toBe('https://fleuret.org/public/lbdl.pdf#page=83');
      expect(entry.license_evidence_url).toBe('https://fleuret.org/public/lbdl.pdf#page=189');
      expect(entry.license).toBe('CC BY-NC-SA 4.0');
      expect(entry.derivation).toBe('pdf-page-render-crop');
      expect(entry.source_page).toBe(83);
      expect(entry.render_dpi).toBe(420);
      expect(entry.crop_box_px).toEqual([61, 42, 1179, 1683]);
      expect(pngDimensions(path)).toEqual([1118, 1641]);
      expect(entry.sha256).toBe('4b6f95a13bb31e5e4f73f2c7d7a6e3270c5cae714d396c804191094692d443db');
    } else {
      const known: Record<string, [number, number, string]> = {
        'uva-activation-std-001.svg': [27, 0, '62cce2bdb87c36b292b81a48a4cd8943af2ed274cbc962a4efb3961982c3e978'],
        'uva-activation-std-01.svg': [29, 0, '6861d8e8ed28675a19c76469b7cf06fddd2ce48bcebefa6ca5a7a27499226099'],
        'uva-activation-fan-in.svg': [32, 1, '8f34fc231bdf4d1dad776001aecfc2d8405bcdd3e551a46216d7f36a9fb47357'],
      };
      expect(known[name], name).toBeDefined();
      const [cell, output, digest] = known[name];
      const pin = 'c7d9e18f42fee53bc8925f08ec2358371b1e9200';
      const repo = 'https://github.com/phlippe/uvadlc_notebooks/blob/' + pin;
      expect(entry.commit).toBe(pin);
      expect(entry.source_url).toBe(repo + '/docs/tutorial_notebooks/tutorial4/Optimization_and_Initialization.ipynb');
      expect(entry.source_asset).toBe(original + 'uva-tutorial4.ipynb');
      expect(entry.source_asset_sha256).toBe('89517153cf67fb903e8a5a18230ae0bc6c027b7d299f53f5181d6d527b422a84');
      expect(entry.source_cell_index).toBe(cell);
      expect(entry.source_output_index).toBe(output);
      expect(entry.source_image_url).toBe('https://uvadlc-notebooks.readthedocs.io/en/latest/_images/tutorial_notebooks_tutorial4_Optimization_and_Initialization_' + cell + '_' + output + '.svg');
      expect(entry.derivation).toBe('notebook-output-extraction');
      expect(entry.crop_box_px).toBeUndefined();
      expect(entry.sha256).toBe(digest);
      expect(entry.license).toBe('MIT');
      expect(entry.license_url).toBe(repo + '/LICENSE.md');
      expect(entry.public_license_copy).toBe('site/public/licenses/uvadlc-c7d9e18f-MIT.txt');
      expect(sha256(resolve(repositoryRoot, entry.public_license_copy!)))
        .toBe('bc760ea73079016ab58ddb9ade0ea6e822dc1ba801b8e26c9d0eadb68e564735');
    }
    expect(sha256(path)).toBe(entry.sha256);
    expectDecodableImage(path);
    // The curated copies and license notice ship; larger source archives do not.
    const source = resolve(repositoryRoot, entry.source_asset!);
    if (existsSync(source)) {
      expect(sha256(source)).toBe(entry.source_asset_sha256);
      if (entry.derivation === 'notebook-output-extraction') {
        const notebook = JSON.parse(readFileSync(source, 'utf8'));
        const svg = notebook.cells[entry.source_cell_index!].outputs[entry.source_output_index!].data['image/svg+xml'];
        expect(readFileSync(path, 'utf8')).toBe(Array.isArray(svg) ? svg.join('') : svg);
      }
    }
    return;
  }

  // Licensed figures can come from tracked paper/assignment/lecture PDFs,
  // including a pinned paper in an Originals packet, or the
  // original raster cache of an executable lecture.
  if (entry.provenance_confirmation === 'source-license-verified'
    && ['pdf-page-render-crop', 'source-raster-copy'].includes(entry.derivation ?? '')) {
    if (entry.derivation === 'pdf-page-render-crop') {
      expect(entry.source_asset).toMatch(/^05 Источники\/(?:Courses\/[^/]+\/(?:Assignments\/[^/]+|Lectures)|Papers\/Source PDFs|Originals\/[^/]+)\/[^/]+\.pdf$/);
      expect(Number.isInteger(entry.source_page) && Number(entry.source_page) > 0).toBe(true);
      requireNonEmptyString(entry.modifications, 'crop description');
    } else if (entry.source_asset === '05 Источники/Originals/ngram-author-figures-2026-11-10/bjc-this-is-a-sentence.png') {
      expect(entry.asset).toBe('00 Учебник/Assets/Figures/curated/ngram-author-figures-2026-11-10/bjc-this-is-a-sentence.png');
      expect(entry.commit).toBe('cb95da139e17ccfb4a4719c15e56bcddcef0e008');
      expect(entry.source_url).toBe('https://bjc.edc.org/bjc-r/sparks/student-pages/U4/L4/04-setting-up-n-grams.html');
      expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/bjc-edc/bjc-r/cb95da139e17ccfb4a4719c15e56bcddcef0e008/sparks/img/U4/lab04/this-is-a-sentence.png');
      expect(entry.source_page_snapshot).toBe('05 Источники/Originals/ngram-author-figures-2026-11-10/bjc-ngram.html');
      expect(sha256(resolve(repositoryRoot, entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
      expect(readFileSync(resolve(repositoryRoot, entry.source_page_snapshot!), 'utf8')).toContain('/bjc-r/sparks/img/U4/lab04/this-is-a-sentence.png');
      expect(entry.license).toBe('CC BY-NC-SA 4.0');
      expect(entry.license_evidence).toBe('05 Источники/Originals/ngram-author-figures-2026-11-10/bjc-sparks.html');
      expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
      expect(readFileSync(resolve(repositoryRoot, entry.license_evidence!), 'utf8')).toContain('rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/"');
      expect(pngDimensions(resolve(repositoryRoot, entry.asset))).toEqual([432,195]);
      expect(entry.crop_box_px).toBeUndefined();
    } else if (entry.source_asset === '05 Источники/Originals/gradient-clipping-2026-10-29/autoclip-mi.gif') {
      const packet = '05 Источники/Originals/gradient-clipping-2026-10-29/';
      expect(entry.asset).toBe('00 Учебник/Assets/Figures/curated/gradient-clipping-2026-10-29/autoclip-mi.gif');
      expect(entry.source_image_url).toBe('https://raw.githubusercontent.com/pseeth/autoclip/dfb622c3caca9dc0ecc8bf0afeb773e190256d6e/images/mi.gif');
      expect(entry.source_asset_sha256).toBe('57cdb8d1dd158320347f4a1017a4554dd0405c40cbaf311e1107c2ec38367397');
      expect(entry.license).toBe('MIT');
      expect(entry.license_evidence).toBe(packet+'autoclip-MIT.txt');
      expect(entry.license_evidence_sha256).toBe('3391e95d6d3cc339dd66a12af61511fbe637abac4fa18254a8c667f794a8fc33');
      expect(sha256(resolve(repositoryRoot,entry.license_evidence!))).toBe(entry.license_evidence_sha256);
      expect(entry.source_page_snapshot).toBe(packet+'autoclip-README.md');
      expect(sha256(resolve(repositoryRoot,entry.source_page_snapshot!))).toBe(entry.source_page_sha256);
    } else if (entry.source_asset?.startsWith('05 Источники/Originals/small-data-finetuning-2026-10-29/')) {
      const packet = '05 Источники/Originals/small-data-finetuning-2026-10-29/';
      const known: Record<string, Partial<CuratedAsset>> = {
        [packet + 'asapp-repo_illustration.png']: {
          commit: '0aa4f4e117ee4422f7cb9355158203e01d6730db', license: 'MIT',
          source_image_url: 'https://raw.githubusercontent.com/asappresearch/revisit-bert-finetuning/0aa4f4e117ee4422f7cb9355158203e01d6730db/repo_illustration.png',
          license_evidence: packet + 'asapp-LICENSE',
          license_evidence_sha256: '573d2a944b2964ab1c030d71f15cdbc18b171534f18457a11b075dbfdcfe039c',
        },
        [packet + 'mosbach-fig1.png']: {
          commit: '3cf27e8667aee9d1c822d747c63ce331b231f283', license: 'Apache-2.0',
          source_image_url: 'https://raw.githubusercontent.com/uds-lsv/bert-stable-fine-tuning/3cf27e8667aee9d1c822d747c63ce331b231f283/images/fig1.png',
          license_evidence: packet + 'mosbach-LICENSE',
          license_evidence_sha256: 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
        },
      };
      const expected = known[entry.source_asset];
      expect(expected).toBeDefined();
      for (const [field, value] of Object.entries(expected))
        expect(entry[field as keyof CuratedAsset]).toBe(value);
      expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    } else if (entry.source_asset?.startsWith('05 Источники/Originals/peft-2026-10-29/')) {
      expect(entry.source_asset).toMatch(/^05 Источники\/Originals\/peft-2026-10-29\/(?:lora-animated\.gif|lora-merge\.png|prompt-tuning\.png|prefix-tuning\.png)$/);
      expect(entry.commit).toBe('861b6c920fe27492d35256c39f266592a518d50c');
      expect(entry.source_image_url).toMatch(/^https:\/\/huggingface\.co\/datasets\/huggingface\/documentation-images\/resolve\/861b6c920fe27492d35256c39f266592a518d50c\/peft\/(?:lora_animated\.gif|lora_diagram\.png|prompt-tuning\.png|prefix-tuning\.png)$/);
      expect(entry.license).toBe('CC BY-NC-SA 4.0');
      expect(entry.license_evidence).toBe('05 Источники/Originals/peft-2026-10-29/hf-images-license.md');
      expect(entry.license_evidence_sha256).toBe('5e6a73b7ff47813e90e2ed21d14807256b2e89fa55e5075ec69fdf01cde801ce');
      expect(sha256(resolve(repositoryRoot, entry.license_evidence!))).toBe(entry.license_evidence_sha256);
    } else {
      expect(entry.source_asset).toMatch(/^05 Источники\/Courses\/[^/]+\/Lectures\/repository\/var\/files\/[^/]+$/);
    }
    expect(entry.source_asset_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.source_asset))).toBe(entry.source_asset_sha256);
    expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256(resolve(repositoryRoot, entry.asset))).toBe(entry.sha256);
    if (entry.derivation === 'source-raster-copy') {
      expect(entry.sha256).toBe(entry.source_asset_sha256);
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
  it('keeps the credited Harvard editions as archived bytes and the replacements attributed', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'),
      { merge: true }) as { assets: CuratedAsset[] };
    const audit = JSON.parse(readFileSync(resolve(root,
      'docs/audit/2026-11-11-harvard-credited-illustrations-additions.json'), 'utf8')) as {
        retired: (CuratedAsset & { bytes: number; git_blob_sha1: string })[];
        figures: CuratedAsset[];
        newFigure: CuratedAsset;
      };
    expect(audit.retired).toHaveLength(6);
    expect(audit.figures).toHaveLength(4);
    for (const spec of audit.retired) {
      expect(registry.assets.find(row => row.asset === spec.asset)).toMatchObject(spec);
      expect(spec.used_in).toEqual([]);
      expect(spec.publication_status).toBe('retired');
      const path = resolve(root, spec.asset);
      const bytes = readFileSync(path);
      expect(bytes.length).toBe(spec.bytes);
      expect(sha256(path)).toBe(spec.sha256);
      expect(createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex'))
        .toBe(spec.git_blob_sha1);
    }
    for (const spec of audit.figures) {
      expect(registry.assets.find(row => row.asset === spec.asset)).toMatchObject(spec);
      const path = resolve(root, spec.asset);
      expectDecodableImage(path);
      expect(sha256(path)).toBe(spec.sha256);
      for (const owner of spec.used_in!) {
        expect(readFileSync(resolve(root, owner), 'utf8')).toContain(spec.asset);
      }
    }
    const evidence = resolve(root, audit.newFigure.license_evidence!);
    expect(audit.newFigure.license_url).toBe('https://creativecommons.org/publicdomain/mark/1.0/');
    if (existsSync(evidence)) {
      expect(sha256(evidence)).toBe(audit.newFigure.license_evidence_sha256);
      expect(readFileSync(evidence, 'utf8')).toContain('Creative Commons Public Domain Mark 1.0');
    }
  });
  it('matches the vLLM source study figures to the same licensed bytes and owners', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'),
      { merge: true }) as { assets: CuratedAsset[] };
    const owner = readFileSync(resolve(root, '05 Источники/Implementation studies/Inside vLLM.md'), 'utf8');
    for (const spec of vllmSourceLibrary.figures) {
      const entry = registry.assets.find(asset => asset.asset === spec.asset);
      expect(entry).toMatchObject(spec);
      expect(sha256(resolve(root, spec.asset))).toBe(spec.sha256);
      expect(owner).toContain(spec.asset + '|');
    }
  });
  it('pins the EAGLE-3 repository original and license in both editions', () => {
    const { assets } = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const row = assets.find(asset => asset.id === 'curated-eagle3-speedup-5e404ac75809');
    expect(row).toBeDefined();
    validateCuratedSourceAsset(row!);
    for (const source of row!.used_in ?? [])
      expect(readFileSync(resolve(root, source), 'utf8')).toContain(row!.asset);
    for (const drift of [{ license: 'CC BY 4.0' }, { crop_box_px: [0, 0, 600, 900] }])
      expect(() => validateCuratedSourceAsset({ ...row!, ...drift })).toThrow();
  });
  it('keeps the five versioned IsoFLOP originals and retires the earlier slide crops', () => {
    const { assets } = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const rows = assets.filter(row => row.asset.startsWith('00 Учебник/Assets/Figures/curated/scaling-isoflop-2026-11-09/'));
    expect(rows).toHaveLength(5);
    const chapter = '00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md';
    const prose = readFileSync(resolve(root, chapter), 'utf8');
    for (const [name, , figure] of isoflopOriginals) {
      const path = isoflopOriginalDir + '/' + name;
      const row = rows.find(asset => asset.asset === path);
      expect(row).toBeDefined();
      validateCuratedSourceAsset(row!);
      for (const owner of row!.used_in!)
        expect(readFileSync(resolve(root, owner), 'utf8')).toContain('![[02 Areas/ML & DL/' + path);
      for (const drift of [{license: 'public access'}, {crop_box_px: [0, 0, 1, 1]}])
        expect(() => validateCuratedSourceAsset({ ...row!, ...drift })).toThrow();
      expect(prose).toContain(path);
      const snapshot = resolve(root, row!.source_page_snapshot!);
      if (existsSync(snapshot)) {
        expect(sha256(snapshot)).toBe(row!.source_page_sha256);
        const html = readFileSync(snapshot, 'utf8');
        expect(html).toContain('License: CC BY 4.0');
        const panel = html.split('<figure id="' + figure.split('#')[1] + '"')[1]?.split('</figure>')[0];
        expect(panel).toContain('/' + name + '"');
      }
    }
    for (const part of ['lower-envelope', 'isoflops', 'joint-fit', 'residuals']) {
      const name = 'chinchilla-' + part + '-chart.png';
      expect(prose).not.toContain(name);
      const old = assets.find(asset => asset.asset.endsWith('/scaling-review-2026-10/' + name));
      expect(old).toMatchObject({ publication_status: 'retired', used_in: [], former_used_in: [chapter] });
      expect(sha256(resolve(root, old!.asset))).toBe(old!.sha256);
    }
  });
  it('keeps the complete Sparse Frontier prefill figure and version-specific license', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const entry = registry.assets.find(row => row.id === 'curated-arithmetic-sparse-frontier-e309c11191da');
    expect(entry).toBeDefined();
    validateCuratedSourceAsset(entry!);
    for (const invalid of [{ license: 'open access' }, { sha256: '0'.repeat(64) },
      { source_image_url: 'https://example.org/other.svg' }, { crop_box_px: [0, 0, 20, 20] }]) {
      expect(() => validateCuratedSourceAsset({ ...entry!, ...invalid })).toThrow();
    }
  });
  it('preserves the complete Google MLCC softmax figure and its page-level rights', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const entry = registry.assets.find(row => row.id === 'curated-google-softmax-multiclass-7b0f01cc61b9')!;
    validateCuratedSourceAsset(entry);
    for (const invalid of [{ license: 'open access' }, { source_image_url: 'https://example.org/img.png' },
      { crop_box_px: [0, 0, 100, 100] }])
      expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
  });
  it('preserves the licensed bagging original and retires only the CS229 raster', () => {
    const { assets } = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const active = assets.find(row => row.id === 'curated-sklearn-tree-bagging-8fa36b8ac57a')!;
    validateCuratedSourceAsset(active);
    for (const drift of [{ license: 'MIT' }, { source_image_url: 'https://example.org/other.png' }, { crop_box_px: [0, 0, 300, 400] }])
      expect(() => validateCuratedSourceAsset({ ...active, ...drift })).toThrow();

    const retired = assets.find(row => row.id === 'curated-cs229-bias-variance-0000000000ac')!;
    validateCuratedSourceAsset(retired);
    expect(() => validateCuratedSourceAsset({ ...retired, license_url: 'https://cs229.stanford.edu/' })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...retired, used_in: active.used_in })).toThrow();
  });
  it('retains both complete MOOC scaling SVGs and rejects changed provenance', () => {
    const prefix = '00 Учебник/Assets/Figures/curated/selection-metric-2026-11-04/';
    const rows = JSON.parse(readFileSync(resolve(root, prefix + 'assets.json'), 'utf8')) as CuratedAsset[];
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(2);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      expect(registry.find(asset => asset.id === row.id)).toEqual(row);
      for (const change of [{id: row.id!.replace('curated-', '')}, {sha256: '0'.repeat(64)},
        {source_asset_sha256: '0'.repeat(64)}, {commit: '0'.repeat(40)}, {license: 'CC0'},
        {license_evidence_sha256: '0'.repeat(64)}, {public_license_copy: 'LICENSE'},
        {source_image_url: 'https://example.org/figure.svg'}, {source_viewbox: [0, 0, 1, 1]},
        {derivation: 'source-svg-crop'}, {crop_box_px: [0, 0, 1, 1]},
        {provenance_confirmation: 'user-confirmed-open-materials'}, {rights_status: 'unverified'}])
        expect(() => validateCuratedSourceAsset({...row, ...change})).toThrow();
    }
  });
  it('preserves the complete py-spy source graph and rejects provenance drift',()=>{
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'),{merge:true}) as {assets:CuratedAsset[]};
    const rows=registry.assets.filter(row=>row.id==='curated-py-spy-flamegraph-592970e72ed1');
    expect(rows).toHaveLength(1);
    validateCuratedSourceAsset(rows[0]);
    for(const invalid of [
      {author:'Unknown'},{license:'public-domain'},{commit:'0'.repeat(40)},
      {source_image_url:'https://example.org/a.svg'},{source_asset:'untracked.svg'},
      {source_asset_sha256:'0'.repeat(64)},{source_page_sha256:'0'.repeat(64)},
      {sha256:'0'.repeat(64)},{license_evidence_sha256:'0'.repeat(64)},
      {derivation:'redraw'},{provenance_confirmation:'unverified'}
    ])expect(()=>validateCuratedSourceAsset({...rows[0],...invalid})).toThrow();
  });
  it('preserves the complete DALI source graph and rejects provenance drift',()=>{
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'),{merge:true}) as {assets:CuratedAsset[]};
    const rows=registry.assets.filter(row=>row.id==='curated-dali-two-readers-3a04306186cc');
    expect(rows).toHaveLength(1);
    validateCuratedSourceAsset(rows[0]);
    for(const invalid of [
      {author:'Unknown'}, {license:'public-domain'}, {commit:'0'.repeat(40)},
      {source_image_url:'https://example.org/a.svg'}, {source_asset:'untracked.svg'},
      {source_asset_sha256:'0'.repeat(64)}, {source_page_sha256:'0'.repeat(64)},
      {sha256:'0'.repeat(64)}, {license_evidence_sha256:'0'.repeat(64)},
      {derivation:'redraw'}, {provenance_confirmation:'unverified'}
    ])expect(()=>validateCuratedSourceAsset({...rows[0],...invalid})).toThrow();
  });
  it('preserves the four former derivative originals and exact native source code', () => {
    const packet = JSON.parse(readFileSync(resolve(root,
      '05 Источники/Originals/mml-source-2026-10-30/retired-derivative-figures.json'), 'utf8'));
    expect(packet.figures).toHaveLength(4);
    for (const row of packet.figures) {
      expect(existsSync(resolve(root, row.from))).toBe(false);
      expect(sha256(resolve(root, row.to))).toBe(row.sha256);
      expect(row.original_registry_record.asset).toBe(row.from);
    }
    const assets = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/derivative-native-2026-10-30/assets.json'), 'utf8'));
    for (const row of assets) {
      expect(sha256(resolve(root, row.source_code))).toBe(row.source_code_sha256);
      expect(sha256(resolve(root, row.license_evidence))).toBe(row.license_evidence_sha256);
    }
  });
  it('preserves the retired MML source privately without treating personal use as a public grant', () => {
    const packet = JSON.parse(readFileSync(resolve(root,
      '05 Источники/Originals/mml-source-2026-10-30/retired-figures.json'), 'utf8'));
    expect(sha256(resolve(root, packet.retained_asset)))
      .toBe('33959e0d35b9d53bec9aa488780a44bb6b5426c39e84818b7c0e5ffc86ba79d1');
    expect(sha256(resolve(root, packet.source_pdf)))
      .toBe('3f87b70c64a35d30ec4e565b4d5cfd18fe913002d3b3bed44691c7d0838911bc');
    expect([packet.pdf_page, packet.printed_page]).toEqual([23, 17]);
    expect(existsSync(resolve(root, packet.original_registry_record.asset))).toBe(false);
  });
  it('binds six newly confirmed legacy figures to exact primary originals and named grants', () => {
    const manifest = JSON.parse(readFileSync(resolve(root, '05 Источники/Originals/publication-asset-provenance-2026-10-29/verified-assets.json'), 'utf8'));
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    expect(manifest.assets).toHaveLength(6);
    for (const spec of manifest.assets) {
      const row = registry.assets.find(entry => entry.asset === spec.asset)!;
      validateCuratedSourceAsset(row);
      for (const drift of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_page_sha256: '0'.repeat(64) }, { license_evidence_sha256: '0'.repeat(64) },
        { source_image_url: 'https://example.invalid/figure.png' }, { license: 'CC0' },
        { provenance_confirmation: 'user-confirmed-open-materials' }, { metadata_status: 'needs-confirmation' },
        { crop_box_px: [0, 0, 10, 10] }])
        expect(() => validateCuratedSourceAsset({ ...row, ...drift })).toThrow();
    }
  });
  it('pins complete CAI and HELM originals to their actual CC BY licenses and canonical chapters', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const ids = ['curated-helm-taxonomy-0000000000c6', 'curated-cai-overview-43db0cfc079e'];
    const rows = registry.assets.filter(row => ids.includes(row.id ?? ''));
    expect(rows).toHaveLength(2);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const drift of [
        { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_page_sha256: '0'.repeat(64) }, { license_evidence_sha256: '0'.repeat(64) },
        { license: 'CC0' }, { license_url: 'https://example.invalid/license' },
        { source_url: row.source_url!.replace(/v\d+$/, '') },
        { source_image_url: 'https://example.invalid/figure.png' },
        { provenance_confirmation: 'user-confirmed-open-materials' },
        { dimensions_px: [10, 10] }, { used_in: ['01 Справочник/Evaluation/Evaluation.md'] },
        { source_asset: '../figure.png' }
      ])
        expect(() => validateCuratedSourceAsset({ ...row, ...drift })).toThrow();
    }
    const helm = rows.find(row => row.id === ids[0])!;
    for (const drift of [{ crop_box_px: [270, 300, 2280, 1250] }, { source_page: 5 }, { render_dpi: 150 }])
      expect(() => validateCuratedSourceAsset({ ...helm, ...drift })).toThrow();
    const cai = rows.find(row => row.id === ids[1])!;
    expect(() => validateCuratedSourceAsset({ ...cai, crop_box_px: [0, 0, 10, 10] })).toThrow();
  });
  it('registers seven complete CS336 residual figures without conflating original figures and screenshots',()=>{
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    const rows=registry.filter(row=>row.asset.startsWith('00 Учебник/Assets/Figures/curated/cs336-residuals-2026-10-29/'));
    expect(rows).toHaveLength(7);
    for(const row of rows){
      validateCuratedSourceAsset(row);
      for(const file of (row as CuratedAsset&{used_in:string[]}).used_in)
        expect(readFileSync(resolve(root,file),'utf8')).toContain(row.asset.split('/').at(-1)!);
      for(const change of [{sha256:'0'.repeat(64)},{license:'CC0'},{source_url:'https://example.org'},{dimensions_px:[1,1]},{rights_status:'unrestricted'}])
        expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
    }
    const gcg=rows.find(row=>row.id?.includes('-gcg-'))!;
    expect(gcg.rights_status).toContain('MIT repository code does not license these screenshots');
    expect(gcg.modifications).toContain('historical montage');
    for(const row of rows.filter(row=>row.id?.includes('-arc')))
      expect(row.rights_status).toContain('limited-educational-quotation');
    const hca=rows.find(row=>row.id?.includes('-hca-'))!;
    expect(hca.derivation).toBe('pdf-figure-extraction');
    expect((hca as unknown as {page:number}).page).toBe(11);
  });
  it('retains the complete Harvard retraining-policy figure as an illustrative source graph',()=>{
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    const row=registry.find(row=>row.id==='curated-serving-decisions-retraining-policies-aff285a0ee3e')!;
    expect(row).toBeDefined();
    validateCuratedSourceAsset(row);
    expect(readFileSync(resolve(root,'05 Источники/Originals/mlops-2026-10-20/harvard-ml-ops.qmd'),'utf8')).toContain('fig-rotting-asset-curve');
    expect(readFileSync(resolve(root,'00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md'),'utf8')).toContain('retraining-policies.png');
    for(const change of [{sha256:'0'.repeat(64)},{source_asset_sha256:'0'.repeat(64)},{license:'MIT'},{derivation:'redraw'},{crop_box_px:[0,0,10,10]}])
      expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
  });
  it('retains both complete Google TPU animations and their page-level CC BY evidence', () => {
    const prefix = '00 Учебник/Assets/Figures/curated/hardware-completion-2026-10-29/';
    const rows = JSON.parse(readFileSync(resolve(root, prefix + 'assets.json'), 'utf8')) as Array<CuratedAsset & {used_in: string[]}>;
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(2);
    // Walk GIF data blocks, not byte occurrences: compressed image data can
    // contain the image-separator byte without beginning a new frame.
    function countFrames(bytes: Buffer): number {
      expect(bytes.subarray(0, 6).toString('ascii')).toMatch(/^GIF8[79]a$/);
      let offset = 13 + ((bytes[10]! & 0x80) ? 3 * 2 ** ((bytes[10]! & 7) + 1) : 0);
      let frames = 0;
      const skipBlocks = () => {
        for (;;) {
          expect(offset).toBeLessThan(bytes.length);
          const size = bytes[offset++]!;
          if (size === 0) return;
          offset += size;
          expect(offset).toBeLessThanOrEqual(bytes.length);
        }
      };
      while (offset < bytes.length) {
        const block = bytes[offset++]!;
        if (block === 0x3b) {
          expect(offset).toBe(bytes.length);
          return frames;
        }
        if (block === 0x21) {
          offset += 1; // extension label
          skipBlocks();
        } else if (block === 0x2c) {
          const packed = bytes[offset + 8]!;
          offset += 9;
          if (packed & 0x80) offset += 3 * 2 ** ((packed & 7) + 1);
          offset += 1; // LZW minimum code size
          skipBlocks();
          frames += 1;
        } else throw new Error('Unexpected GIF block at ' + (offset - 1));
      }
      throw new Error('GIF trailer missing');
    }
    rows.forEach((row, index) => {
      validateCuratedSourceAsset(row);
      expect(registry.find(asset => asset.id === row.id)).toEqual(row);
      for (const owner of row.used_in) {
        const chapter = readFileSync(resolve(root, owner), 'utf8');
        expect(chapter).toContain(row.asset + '|');
        expect(chapter).toContain(row.source_url);
      }
      expect(row.license).toBe('CC BY 4.0');
      expect(row.source_url).toBe('https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm');
      expect(row.source_image_url).toBe('https://docs.cloud.google.com/static/tpu/docs/images/' +
        ['image4_5pfb45w.gif', 'image1_2pdcvle.gif'][index]);
      expect(row.dimensions_px).toEqual([640, 269]);
      expect(row.frames).toBe([103, 267][index]);
      expect(countFrames(readFileSync(resolve(root, row.asset)))).toBe(row.frames);
      const snapshot = readFileSync(resolve(root, row.source_page_snapshot!), 'utf8');
      expect(snapshot).toContain(row.source_image_url!.replace('https://docs.cloud.google.com', ''));
      expect(snapshot).toContain('https://creativecommons.org/licenses/by/4.0/');
      expect(snapshot).toContain('Except as otherwise noted, the content of this page is licensed');
      for (const path of row.used_in) expect(readFileSync(resolve(root, path), 'utf8')).toContain(row.asset.split('/').at(-1)!);
      for (const change of [{sha256:'0'.repeat(64)}, {source_asset_sha256:'0'.repeat(64)},
        {source_page_sha256:'0'.repeat(64)}, {license_evidence_sha256:'0'.repeat(64)},
        {license:'Apache-2.0'}, {frames:1}, {crop_box_px:[0,0,20,20]}])
        expect(() => validateCuratedSourceAsset({...row,...change})).toThrow();
    });
  });

  it('registers complete Harvard serving-operation SVGs with their source QMD and license', () => {
    const prefix = '00 Учебник/Assets/Figures/curated/serving-operations-2026-10-29/';
    const rows = JSON.parse(readFileSync(resolve(root, prefix + 'assets.json'), 'utf8')) as Array<CuratedAsset & {used_in:string[]}>;
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(2);
    rows.forEach((row,index) => {
      validateCuratedSourceAsset(row);
      expect(registry.find(asset => asset.id === row.id)).toEqual(row);
      expect(row.derivation).toBe('source-svg-copy');
      expect(row.license).toBe('CC BY-NC-SA 4.0');
      expect(row.commit).toBe('2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd');
      const source = readFileSync(resolve(root, row.source_page_snapshot!), 'utf8');
      expect(source).toContain(['#fig-power-diagram', '#fig-ops-layers'][index]);
      for (const path of row.used_in) expect(readFileSync(resolve(root,path),'utf8')).toContain(row.asset.split('/').at(-1)!);
      for (const change of [{sha256:'0'.repeat(64)}, {source_asset_sha256:'0'.repeat(64)},
        {source_page_sha256:'0'.repeat(64)}, {license_evidence_sha256:'0'.repeat(64)},
        {license:'CC BY 4.0'}, {derivation:'source-svg-crop'}, {crop_box_px:[0,0,20,20]}])
        expect(() => validateCuratedSourceAsset({...row,...change})).toThrow();
    });
  });
  it('rejects changed source, license, pixels or crop metadata for original AD traces',()=>{
    const rows=JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/autodiff-2026-10-27/assets.json'),'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(3);
    for(const row of rows) {
      validateCuratedSourceAsset(row);
      for(const change of [{sha256:'0'.repeat(64)},{source_asset_sha256:'0'.repeat(64)},
        {license:'CC0'},{render_dpi:100},{crop_box_px:[0,0,10,10]},
        {source_url:'https://example.org/figure.pdf'},{source_page:1}])
        expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
    }
  });
  it('keeps the complete pinned vLLM interface diagram and rejects license/hash drift', () => {
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    const row=registry.find(r=>r.asset.includes('/http-lifecycle-2026-10-26/'))!;
    expect(row).toBeTruthy();
    validateCuratedSourceAsset(row);
    for(const change of [{license:'MIT'},{source_asset_sha256:'0'.repeat(64)},
      {sha256:'0'.repeat(64)},{crop_box_px:[0,0,1,1]},{author:''}])
      expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
  });
  it('preserves four complete original UvA SVG notebook outputs and rejects provenance drift',()=>{
    const rows=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets
      .filter((x:CuratedAsset)=>x.asset.includes('/autoencoder-course-2026-11-03/')) as CuratedAsset[];
    expect(rows).toHaveLength(4);
    for(const row of rows){
      validateCuratedSourceAsset(row);
      for(const change of [{license:'CC0'},{sha256:'0'.repeat(64)},
        {source_cell:0},{source_member_sha256:'0'.repeat(64)},
        {source_asset_sha256:'0'.repeat(64)},{source_url:'https://example.org'},
        {crop_box_px:[0,0,1,1]},{provenance_confirmation:'user-confirmed-open-materials'}])
        expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
    }
  });
  it('preserves original AE source SVG and all GIF frames with pinned source permissions', () => {
    const prefix = '00 Учебник/Assets/Figures/curated/autoencoder-2026-10-26/';
    const rows = JSON.parse(readFileSync(resolve(root, prefix + 'assets.json'), 'utf8')) as CuratedAsset[];
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(2);
    const hashes = ['19f02e477d2cb875ecb7705730fbe896f33497353bd6d68f15ef99296c859c97',
      '9b41abceb5b1d151d1f9cf1902eca8aebfbe6bfac5638f4526c5b5328088a7d5'];
    rows.forEach((row, i) => {
      validateCuratedSourceAsset(row);
      expect(row.sha256).toBe(hashes[i]);
      expect(row.license).toBe('MIT');
      expect(row.derivation).toBe(i ? 'source-gif-copy' : 'source-svg-copy');
      expect(registry.find(entry => entry.id === row.id)).toEqual(row);
      for (const change of [{sha256:'0'.repeat(64)}, {license:'CC0'},
        {source_page_sha256:'0'.repeat(64)}, {crop_box_px:[0,0,5,5]}, {frames:1}])
        expect(() => validateCuratedSourceAsset({...row,...change})).toThrow();
    });
    // A one-frame thumbnail must not silently replace the 25-frame source animation.
    expect(rows[1].frames).toBe(25);
    for (const suffix of ['weng-autoencoder.png', 'weng-vae-graphical-model.png', 'weng-reparameterization.png']) {
      const row = registry.find(entry => entry.asset.endsWith('/'+suffix))!;
      expect(row.metadata_status).toBe('needs-confirmation');
      expect(row.rights_status).toBe('permission-unconfirmed');
      expect(row.license).not.toBe('MIT');
      expect(row.publication_status).toBe('retired');
      expect(row.used_in).toEqual([]);
    }
    expect(registry.find(entry=>entry.asset.endsWith('/weng-reparameterization.png'))!.author).toContain('Kingma');
  });
  it('preserves the four complete CNN source SVGs and their pinned provenance', () => {
    const prefix = '00 Учебник/Assets/Figures/curated/cnn-2026-10-26/';
    const rows = JSON.parse(readFileSync(resolve(root, prefix + 'assets.json'), 'utf8')) as CuratedAsset[];
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets as CuratedAsset[];
    const hashes = ['f75f5ed4cdc666d5813ea08ac2e81b5a8796d67e91efd7071db7d412565a1d22',
      '2eaab1b9c265704fdd1f2ab3b0ebc7c9188155f0a9dfcbfb4feb2d45be27bf44',
      '12643b8bcd701e3ace01ab0726e674b967771a297ee34629f2911e5a84e1612c',
      'c0fb4a2104bb382aa00caa3b2f3ab18435724618259e23624335e325ec5568c3'];
    expect(rows).toHaveLength(4);
    rows.forEach((row, index) => {
      validateCuratedSourceAsset(row);
      expect(row.sha256).toBe(hashes[index]);
      expect(row.commit).toBe('23d7a5aecceee57d1292c56e90cce307f183bb0a');
      expect(row.license).toBe('CC BY-SA 4.0');
      expect(row.derivation).toBe('source-svg-copy');
      expect(registry.find(entry => entry.id === row.id)).toEqual(row);
      for (const change of [{sha256: '0'.repeat(64)}, {license: 'CC0'},
        {source_page_sha256: '0'.repeat(64)}, {crop_box_px: [0,0,5,5]}])
        expect(() => validateCuratedSourceAsset({...row, ...change})).toThrow();
    });
    for (const suffix of ['cs231n-local-connectivity.jpeg', 'cs231n-stride-padding.jpeg']) {
      const row = registry.find(entry => entry.asset.endsWith('/' + suffix))!;
      // These two original note diagrams are byte-matched to the pinned MIT
      // repository. The separate third-party dropout figure stays unverified.
      validateCuratedSourceAsset(row);
      expect(row.metadata_status).toBe('verified');
      expect(row.license).toBe('MIT');
      expect(row.license_evidence).toBe('05 Источники/Originals/publication-asset-provenance-2026-10-29/cs231n-LICENSE.txt');
      for (const drift of [{ license: 'CC BY 4.0' }, { source_asset_sha256: '0'.repeat(64) },
        { license_evidence_sha256: '0'.repeat(64) }, { commit: 'main' }])
        expect(() => validateCuratedSourceAsset({ ...row, ...drift })).toThrow();
    }
  });
  it('credits the original dropout paper without transferring the CS231n code license', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'));
    const figure = registry.assets.find((row: CuratedAsset) => row.asset.endsWith('/cs231n-dropout-train.jpeg'));
    expect(figure.author).toContain('Nitish Srivastava');
    expect(figure.source_url).toBe('https://jmlr.org/papers/v15/srivastava14a.html');
    expect(figure.license).not.toBe('MIT');
    expect(figure.metadata_status).toBe('needs-confirmation');
    expect(figure.publication_status).toBe('retired');
    expect(figure.used_in).toEqual([]);
    // Legacy raw is ignored and absent from public checkouts; preserve tracked artwork bytes.
    expect(sha256(resolve(root, figure.asset))).toBe('9c5d66c1cfa60f53b7cd36454f2a600b8c5cb82665d16157ba79400090331009');
  });
  it('pins complete author figures for dropout and AE/VAE with distinct reuse licenses', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets as CuratedAsset[];
    const rows = registry.filter(entry => entry.asset.startsWith(foundationsFigurePath));
    expect(rows.map(row => row.asset.slice(foundationsFigurePath.length)).sort())
      .toEqual([...Object.keys(princeFigures), 'chervinskii-autoencoder.png'].sort());
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const alteration of [
        { crop_box_px: [0, 0, 1, 1] },
        { sha256: '0'.repeat(64) },
        { license: 'MIT' },
        { source_archive_member: '../wrong.pdf', source_member: '../wrong.pdf' },
      ]) expect(() => validateCuratedSourceAsset({ ...row, ...alteration })).toThrow();
    }
  });
  it('preserves the complete original SAR and weighted-objectives SVGs with source licenses', () => {
    const rows=JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/recommender-mf-2026-10-25/assets.json'),'utf8')) as CuratedAsset[];
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(2);
    const hashes=['10ecfb8f5fa6b62cbea24d5eca145197fddaa584b11cf0c6c1e7ddc10d71e6f6',
      'eb5ce3abf87e0cb4ce5628aff00bf96f25032bc6cd4c86accc4f1871d12f069e'];
    rows.forEach((row,index)=>{
      validateCuratedSourceAsset(row);
      expect(row.sha256).toBe(hashes[index]);
      expect(row.license).toBe(index===0?'MIT':'CC BY 4.0');
      expect(row.derivation).toBe('source-svg-copy');
      expect(row.crop_box_px).toBeUndefined();
      expect(registry.find(entry=>entry.id===row.id)).toEqual(row);
      for(const change of [{sha256:'0'.repeat(64)},{license:'CC0'},
        {source_page_sha256:'0'.repeat(64)},{crop_box_px:[0,0,5,5]}])
        expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
    });
    expect(rows[0].commit).toBe('e76bb39727103e45dd8011b6d0eca2f1aad3c0f7');
    expect(readFileSync(resolve(root,rows[0].asset),'utf8')).toContain('viewBox="0 0 1280 720"');
    expect(readFileSync(resolve(root,rows[1].asset),'utf8')).toContain('viewBox="0.0 0.0 805.6089238845144 275.6430446194226"');
    expect(readFileSync(resolve(root,'site/public/licenses/recommenders-resources-e76bb397-MIT.txt'),'utf8'))
      .toContain('Copyright (c) 2024-present Recommenders contributors.');
  });
  it('preserves four Feast figures and the complete Google recommendation process SVG', () => {
    const rows=JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/recommender-data-2026-10-25/assets.json'),'utf8')) as CuratedAsset[];
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(5);
    const hashes=['303f80696e6e664998393d3696384879b202824e3e6d69930bf5355cf44b9ab1',
      '4898b3c2620dc0d7bb4f5f995c14f1d0a00efe57ee01c4c7a9908087cffdec52',
      '662c258ed5bd3d815ab3ad9426e9b10a0e161660a9bc04f85aa4492f29a37771',
      '10d9dbdcd218a0b40aa8938e0c35744c906d91c092e530e56a733baedcfe1ef8',
      'c4bd0ec086f9b60816dca78e4344077529db1f41dba474df58973716c71e97ab'];
    rows.forEach((row,index)=>{
      validateCuratedSourceAsset(row);
      expect(row.sha256).toBe(hashes[index]);
      expect(row.source_asset_sha256).toBe(hashes[index]);
      expect(row.license).toBe(index<4?'Apache-2.0':'CC BY 4.0');
      if(index<4) {
        expect(row.commit).toBe('1e5e90ca01b0a6b1877a6c809436f0d8d574aad6');
        expect(row.source_page_sha256).toBe('ce3d3a52023509f3a985e2900689835550537303b6e7a72bdcc0cda5c06507a8');
      } else {
        expect(row.source_page_sha256).toBe('035f5beef49f2d199bc1864dcb1444d4a9a4d0529c034e1c491488a3fd82c32d');
        const svg=readFileSync(resolve(root,row.asset),'utf8');
        expect(svg).toContain('viewBox="0.0 0.0 473.79265091863516 206.29921259842519"');
        const page=readFileSync(resolve(root,row.source_page_snapshot!),'utf8');
        expect(page).toContain('/machine-learning/recommendation/images/Process.svg');
        expect(page).toContain('https://creativecommons.org/licenses/by/4.0/');
      }
      expect(registry.find(entry=>entry.id===row.id)).toEqual(row);
      for(const change of [{sha256:'0'.repeat(64)},{license:'CC0'},
        {source_asset_sha256:'0'.repeat(64)},{crop_box_px:[0,0,10,10]}])
        expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
    });
    const license=readFileSync(resolve(root,'site/public/licenses/feast-1e5e90ca-Apache-2.0.txt'),'utf8');
    expect(license).toContain('Apache License');
    expect(license).toContain('Version 2.0');
    const notice=readFileSync(resolve(root,'site/public/licenses/recommender-data-NOTICE.txt'),'utf8');
    expect(notice).toContain('Feast contributors');
    expect(notice).toContain('https://creativecommons.org/licenses/by/4.0/');
  });
  it('preserves seven original SVM figures, exact saved notebook outputs and source license', () => {
    const rows=JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/kernels-svm-rff-2026-10-25/assets.json'),'utf8')) as CuratedAsset[];
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(7);
    const hashes=[
      'ad92ac56e5c63ff9882ca47ff0b15afda601579b30789e673e7bde9a7778512b',
      'f2c0e3b83a9eabc96fbc0d6941f36b26f433ed03c84a382ef1be0f6705e82fae',
      '49d339755fb385a939840d369f908c9d85245d89d13d37ab57612d6ee57efc12',
      '515465b2768e048d454bcc3b59c5fce2ad1394ed706c4f7ef938d230d0be63e6',
      'caa84634bd61dc8f282efd90f28a8917a1c9e0c99e197d578a249e8522dd330b',
      '599d45c8b4fd10b852697243fac9825ce3bedde97dfbc53f0d987c90a1d3c8cb',
      '061db8c3c848d551e828efb17fff702fcf1aa2be21cde3e1be658b6d4ca29730'];
    rows.forEach((row,index)=>{
      validateCuratedSourceAsset(row);
      expect(row.sha256).toBe(hashes[index]);
      expect(row.source_asset_sha256).toBe('8f01bffbe7f86c160fcbe89f4549b1650d262adc0e76b354719c41eec15af0a6');
      expect(row.source_member).toBe(`/cells/${[7,16,20,25,29,33,38][index]}/outputs/0/data/image~1png`);
      expect(row.license).toBe('CC BY-NC-ND 3.0 Unported');
      expect(row.commit).toBe('d66231454ef753818dc9213c9b5942e067266966');
      expect(row.crop_box_px).toBeUndefined();
      expect(registry.find(entry=>entry.id===row.id)).toEqual(row);
      for(const change of [{sha256:'0'.repeat(64)},{license:'CC0'},
        {source_member:'/cells/1/outputs/0/data/image~1png'},
        {source_asset_sha256:'0'.repeat(64)}, {crop_box_px:[0,0,20,20]}])
        expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
    });
    const notice=readFileSync(resolve(root,'site/public/licenses/kernels-visuals-NOTICE.txt'),'utf8');
    expect(notice).toContain('Jake VanderPlas');
    expect(notice).toContain('https://creativecommons.org/licenses/by-nc-nd/3.0/');
  });
  it('preserves both complete original RBF parameter grids and their source protocol',()=>{
    const rows=JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/kernels-2026-11-03/assets.json'),'utf8')) as CuratedAsset[];
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(2);
    for(const row of rows){
      validateCuratedSourceAsset(row);
      expect(registry.find(entry=>entry.id===row.id)).toEqual(row);
      for(const change of [{sha256:'0'.repeat(64)},{license:'CC0'},{commit:'unreviewed'},
        {source_asset_sha256:'0'.repeat(64)},{source_page_sha256:'0'.repeat(64)},
        {license_evidence_sha256:'0'.repeat(64)},{crop_box_px:[0,0,20,20]},
        {source_image_url:'https://example.invalid/figure'}, {dimensions_px:[10,10]}])
        expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
    }
    const notice=readFileSync(resolve(root,'site/public/licenses/kernels-visuals-NOTICE.txt'),'utf8');
    expect(notice).toContain('RBF SVM parameters');
    expect(notice).toContain('scikit-learn-1.9.1-BSD-3-Clause.txt');
    expect(notice).toContain('199d116ab37beed506c63fc9f6a74f56e228de43');
  });
  it('preserves five full GMM generation and initialization originals',()=>{
    const rows=JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/gmm-2026-11-03/assets.json'),'utf8')) as CuratedAsset[];
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(5);
    for(const row of rows){
      validateCuratedSourceAsset(row);
      expect(registry.find(entry=>entry.id===row.id)).toEqual(row);
      for(const change of [{sha256:'0'.repeat(64)},{license:'CC0'},{commit:'unreviewed'},
        {source_asset_sha256:'0'.repeat(64)},{crop_box_px:[0,0,20,20]},
        {source_image_url:'https://example.invalid/figure'}])
        expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
    }
    const notice=readFileSync(resolve(root,'site/public/licenses/gmm-visuals-NOTICE.txt'),'utf8');
    expect(notice).toContain('cells 34, 42, 46 and 52');
    expect(notice).toContain('199d116ab37beed506c63fc9f6a74f56e228de43');
    expect(notice).toContain('scikit-learn-1.9.1-BSD-3-Clause.txt');
  });
  it('preserves six unmodified Gaussian-mixture originals and their exact notebook output pointers', () => {
    const rows=JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/gmm-2026-10-25/assets.json'),'utf8')) as CuratedAsset[];
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(6);
    const hashes=[
      'cf8e9ca21f932fd09a8630312420ca254651bc051eb11e630e54f289b73586bd',
      'a2d1b195371fed3988a9e1cdd15f29928f945ab6dbfc94970a1d94e1a5937f2d',
      'e0edbdb364f2314994113177d0d33178ed613facf4c0ed4e333c301a706f6d6a',
      '6b86261437e82ef1fff5503402a083f0c3247e2c669376a72726faf877b90eea',
      '10bdcdec749e1275783b4f5ccd7e0bf74c1edb099f7dc4f6b36fb0a217afe3ce',
      '1e1f676f674f3296e3ee59c56f76621a23cf2bab292543452ebbe0016ab947b3'];
    rows.forEach((row,index)=>{
      validateCuratedSourceAsset(row);
      expect(row.sha256).toBe(hashes[index]);
      expect(row.license).toBe('CC BY-NC-ND 3.0 Unported');
      expect(row.commit).toBe('d66231454ef753818dc9213c9b5942e067266966');
      expect(row.crop_box_px).toBeUndefined();
      expect(registry.find(entry=>entry.id===row.id)).toEqual(row);
      if(index<5) expect(row.source_member).toBe(`/cells/${[17,23,30,32,37][index]}/outputs/0/data/image~1png`);
      for(const change of [{sha256:'0'.repeat(64)},{license:'CC0'},
        {source_member:'/cells/1/outputs/0/data/image~1png'},
        {source_asset_sha256:'0'.repeat(64)}, {crop_box_px:[0,0,20,20]}])
        expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
    });
    const notice=readFileSync(resolve(root,'site/public/licenses/gmm-visuals-NOTICE.txt'),'utf8');
    expect(notice).toContain('Jake VanderPlas');
    expect(notice).toContain('https://creativecommons.org/licenses/by-nc-nd/3.0/');
  });
  it('preserves seven complete PCA/t-SNE/UMAP originals, notebook output and source permissions', () => {
    const rows=JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/pca-tsne-umap-2026-10-25/assets.json'),'utf8')) as CuratedAsset[];
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(7);
    const hashes=[
      '0705fa33bf4c3edf7bb3141dcd82ebfa35897e2bf6886cf74bfa59f35e4843da',
      'f3ef65e6c2adc29869cbbf51f4cc7e844daa1d7eda669f5d3d2221782e9c98e7',
      '3863ed3fefd9aac427be9d4f524cdfaeb2db133f31f46fd7217da36d14b5fcc8',
      '43ab675d0bd6c5f401f020980be57664d3db6d05b0bb5ee0a475471816a93496',
      '950748fe26cf1cfe707e1f33119895f1f267d4aac1aef70ca1fd4470e885611c',
      'b0e2ed3673ef8f8d12ad5304dd8fdf172adfe64b08433527190d0be676c9a4fd',
      'd9e83915a97b8b6da4eec03f3de2e9ba3f1efb74ee09f71a00b07515994a5d3c'];
    rows.forEach((row,index)=>{
      validateCuratedSourceAsset(row);
      expect(row.sha256).toBe(hashes[index]);
      expect(row.crop_box_px).toBeUndefined();
      expect(row.license).toBe(index<2 ? 'CC BY-NC-ND 3.0 Unported' : index<4 ? 'CC BY 2.0' : 'BSD-3-Clause');
      expect(row.commit).toBe(index<2 ? 'd66231454ef753818dc9213c9b5942e067266966' : index<4
        ? 'f2492b469bec8aaa7aa83cc828081e14a295414d' : 'e82ed0d457b566b043ef44f4007a7149b0daca74');
      expect(registry.find(entry=>entry.id===row.id)).toEqual(row);
      for(const change of [{sha256:'0'.repeat(64)}, {license:'CC0'},
        {source_asset_sha256:'0'.repeat(64)}, {crop_box_px:[0,0,50,50]},
        {source_url:'https://example.invalid/figure'}, {dimensions_px:[99,99]}])
        expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
    });
    const archive='05 Источники/Originals/pca-tsne-umap-2026-10-25/';
    const appendix=readFileSync(resolve(root,archive+'jake-figure-code.html'),'utf8');
    expect(appendix).toMatch(/whiten<\/span>.*True/);
    expect(readFileSync(resolve(root,archive+'distill-license-evidence.md'),'utf8')).toContain('CC-BY 2.0');
    const notice=readFileSync(resolve(root,'site/public/licenses/dimensionality-reduction-NOTICE.txt'),'utf8');
    expect(notice).toContain(readFileSync(resolve(root,archive+'umap-LICENSE.txt'),'utf8').trim());
    expect(notice).toContain('https://creativecommons.org/licenses/by-nc-nd/3.0/');
  });
  it('preserves fifteen complete original PCA, Distill and UMAP additions', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/dimensionality-2026-11-03/assets.json'), 'utf8')) as CuratedAsset[];
    const registry = parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(15);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      expect(registry.find(entry => entry.id === row.id)).toEqual(row);
      for (const change of [{sha256:'0'.repeat(64)}, {license:'CC0'},
        {source_asset_sha256:'0'.repeat(64)}, {crop_box_px:[0,0,20,20]},
        {source_image_url:'https://example.invalid/image.png'}, {commit:'unreviewed'}])
        expect(() => validateCuratedSourceAsset({...row,...change})).toThrow();
    }
  });
  it('preserves the connected heat and weighted-constraint figures with exact provenance', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/spectral-2026-10-31/assets.json'), 'utf8')) as CuratedAsset[];
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(2);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      expect(registry.find(entry => entry.id === row.id)).toEqual(row);
      for (const change of [{sha256:'0'.repeat(64)}, {license:'CC0'},
        {source_page:6}, {source_figure:'1.2'}, {crop_box_px:[0,0,20,20]},
        {source_asset_sha256:'0'.repeat(64)}, {render_dpi:144},
        {source_url:'https://example.invalid/figure'}])
        expect(() => validateCuratedSourceAsset({...row, ...change})).toThrow();
    }
    const notice = readFileSync(resolve(root, 'site/public/licenses/spectral-visuals-NOTICE.txt'), 'utf8');
    expect(notice).toContain('wiskott-connected-heat.png (Figure 1.1, PDF page 5)');
    expect(notice).toContain('wiskott-weighted-constraint.png (Figure 2.2, PDF page 15)');
  });
  it('preserves the complete spectral-clustering source figure and unmodified segmentation originals', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/spectral-2026-10-25/assets.json'), 'utf8')) as CuratedAsset[];
    const registry = parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(3);
    const expectedHashes = [
      '8ba2e63a603f9b0f7b3f8b51014ae446dcd2c0b806b61c32411dbfc9ea166851',
      'f279cd6f3797bf8fe9de0ac9cc1634831d27bdd76c033b036006e05de75f17bd',
      '4eb8fde44674086e61aad5d2a96119eadf094764b6af4bc76b1373a12de25b13',
    ];
    rows.forEach((row,index) => {
      validateCuratedSourceAsset(row);
      for(const change of [{sha256:'0'.repeat(64)},{license:'CC0'},
        {source_asset_sha256:'0'.repeat(64)},{crop_box_px:[0,0,20,20]},
        {source_url:'https://example.invalid/figure'}])
        expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
      expect(row.sha256).toBe(expectedHashes[index]);
      expect(sha256(resolve(root,row.asset))).toBe(row.sha256);
      expect(validateRenderedPngBytes(readFileSync(resolve(root,row.asset))))
        .toEqual(index===0 ? [2000,910] : [1000,500]);
      expect(sha256(resolve(root,row.source_asset!))).toBe(row.source_asset_sha256);
      expect(sha256(resolve(root,row.license_evidence!))).toBe(row.license_evidence_sha256);
      expect(registry.find(entry=>entry.id===row.id)).toEqual(row);
      if(index===0) {
        expect(row.license).toBe('CC BY-SA 4.0');
        expect(row.source_page).toBe(6);
        expect(row.render_dpi).toBe(300);
        expect(row.crop_box_px).toEqual([280,710,2280,1620]);
        expect(row.render_page_size_px).toEqual([2550,3300]);
      } else {
        expect(row.license).toBe('BSD-3-Clause');
        expect(row.commit).toBe('866c0f51e7560ef0303cbcc5f159df5382ea9e3f');
        expect(row.crop_box_px).toBeUndefined();
        expect(sha256(resolve(root,row.source_code!))).toBe(row.source_code_sha256);
      }
    });
    const notice=readFileSync(resolve(root,'site/public/licenses/spectral-visuals-NOTICE.txt'),'utf8');
    expect(notice).toContain('Laurenz Wiskott and Fabian Schönfeld');
    expect(notice).toContain(readFileSync(resolve(root,'05 Источники/Originals/spectral-2026-10-25/sklearn-COPYING'),'utf8').trim());
  });
  it('retains both complete HH calibration panels with the pinned source, caption and crop', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/reward-modeling-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(1);
    const row = rows[0];
    validateCuratedSourceAsset(row);
    for (const change of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
      { license: 'CC0' }, { source_url: 'https://arxiv.org/pdf/2204.05862' },
      { crop_box_px: [350, 300, 2200, 850] }, { license_evidence_sha256: '0'.repeat(64) }])
      expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
  });
  it('preserves the three complete HH figures and a pinned MIT dataset excerpt', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/preference-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(3);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const change of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { license: 'CC0' }, { source_url: 'https://arxiv.org/pdf/2204.05862' },
        { crop_box_px: [0, 0, 10, 10] }, { license_evidence_sha256: '0'.repeat(64) }])
        expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
    }
    expect(sha256(resolve(root, '05 Источники/Originals/preference-2026-10-23/hh-helpful-test-line1.jsonl')))
      .toBe('e002aa2fc2e24502a26789d19fb29bb2ad4c27ea2007dc075eaccc283995a806');
    expect(sha256(resolve(root, '05 Источники/Originals/preference-2026-10-23/hh-MIT-LICENSE.txt')))
      .toBe('d4fb18db48757e273261a5597b8a09381de8073da060474ce33d0643e06c875e');
  });
  it('retains the full version-pinned Tulu 3 overview and its paper license evidence', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/sft-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(2);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const change of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { license: 'CC0' }, { crop_box_px: [0, 0, 10, 10] },
        { source_url: 'https://arxiv.org/html/2411.15124' },
        { license_evidence_sha256: '0'.repeat(64) }])
        expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
    }
  });
  it('keeps Harvard rendered originals distinct from their pinned generating source', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/fault-tolerance-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(2);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const change of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { license: 'CC0' }, { crop_box_px: [0, 0, 10, 10] }, { commit: '0'.repeat(40) },
        { source_image_url: 'https://example.invalid/chart.png' }])
        expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
    }
  });
  it('retains full source network/UCP diagrams with distinct MIT and Apache notices', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/network-checkpoint-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(2);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const change of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { license: 'CC0' }, { crop_box_px: [0, 0, 10, 10] }, { commit: '0'.repeat(40) },
        { source_image_url: row.source_image_url!.replace(row.commit!, 'main') }])
        expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
    }
  });
  it('preserves the complete Playbook EP layer and its pinned Apache evidence', () => {
    const [row] = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/expert-hybrid-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    validateCuratedSourceAsset(row);
    for (const change of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
      { license: 'CC0' }, { crop_box_px: [0, 0, 10, 10] }, { commit: '0'.repeat(40) },
      { source_image_url: row.source_image_url!.replace(row.commit!, 'main') }])
      expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
  });
  it('retains full Cairo renders of three pinned FSDP SVGs with equations and source provenance', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/fsdp-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(3);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const change of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_code_sha256: '0'.repeat(64) }, { license: 'CC0' },
        { crop_box_px: [0, 0, 10, 10] }, { commit: '0'.repeat(40) }])
        expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
    }
  });
  it('retains complete RLVR test panels, caption and original reward-mixing graph', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/rlvr-2026-10-24/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(2);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      expect(row.license).toBe('CC BY 4.0');
      expect(row.author).toBe('Nathan Lambert et al.');
      expect(sha256(resolve(root, row.license_evidence!))).toBe(row.license_evidence_sha256);
      expectDecodableImage(resolve(root, row.asset));
      for (const change of [{sha256:'0'.repeat(64)}, {source_asset_sha256:'0'.repeat(64)}])
        expect(() => validateCuratedSourceAsset({...row,...change})).toThrow();
    }
    expect(rows[0].source_page).toBe(33);
    expect(rows[0].render_dpi).toBe(400);
    expect(rows[0].crop_box_px).toEqual([380,350,3030,3190]);
    expect(pngDimensions(resolve(root,rows[0].asset))).toEqual([2650,2840]);
    expect(rows[0].sha256).toBe('f5480a3783fbc5c60bc3de15c4490d327184576f7eb848233c510f20f9ba3b4e');
    expect(rows[1].source_image_url).toBe('https://arxiv.org/html/2411.15124v5/figures_images/post-training/x32_RLVR_KL1.png');
    expect(rows[1].crop_box_px).toBeUndefined();
    expect(pngDimensions(resolve(root,rows[1].asset))).toEqual([1886,1320]);
    expect(rows[1].sha256).toBe('468af70e83892da0ae7a3ed33eecb8a79d75533e95e0bd4467781c594d2b2ef8');
  });
  it('retains full Tulu preference/ablation figures with captions from the existing licensed v5 PDF', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/dpo-2026-10-24/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(2);
    rows.forEach((row, i) => {
      validateCuratedSourceAsset(row);
      expect(row.source_asset).toBe('05 Источники/Papers/Source PDFs/tulu3-rlvr-arxiv-v5.pdf');
      expect(row.source_asset_sha256).toBe('79bed1ef4ce4bb332f43bdc87d60709e644560f8f3bb6d14913a473cc82c251f');
      expect(row.source_page).toBe([22,25][i]);
      expect(row.render_dpi).toBe(400);
      expect(row.crop_box_px).toEqual([[333,306,3056,1278],[361,1083,3055,2317]][i]);
      expect(pngDimensions(resolve(root, row.asset))).toEqual([[2723,972],[2694,1234]][i]);
      expect(row.license).toBe('CC BY 4.0');
      expect(sha256(resolve(root, row.license_evidence!))).toBe(row.license_evidence_sha256);
      for (const change of [{sha256:'0'.repeat(64)}, {source_asset_sha256:'0'.repeat(64)}])
        expect(() => validateCuratedSourceAsset({...row,...change})).toThrow();
    });
  });
  it('retains the complete pinned PPO framework without inventing a duplicate source archive', () => {
    const [row] = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/ppo-2026-10-24/assets.json'), 'utf8')) as CuratedAsset[];
    validateCuratedSourceAsset(row);
    for (const change of [{ sha256: '0'.repeat(64) }, { license: 'CC0' },
      { crop_box_px: [0, 0, 10, 10] }, { commit: '0'.repeat(40) },
      { source_image_url: 'https://example.org/image.png' }])
      expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
  });
  it('preserves all five complete author-published Zero Bubble figures and the Space license evidence', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/zero-bubble-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(5);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const change of [{ sha256: '0'.repeat(64) }, { license: 'CC0' },
        { crop_box_px: [0, 0, 10, 10] }, { commit: '0'.repeat(40) },
        { source_image_url: 'https://example.org/image.png' },
        { source_url: row.source_url!.replace(row.commit!, 'main') }])
        expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
    }
    const license = readFileSync(resolve(root, 'site/public/licenses/zero-bubble-Apache-2.0.txt'), 'utf8');
    expect(license).toContain('Apache License');
    expect(license).toContain('Version 2.0, January 2004');
  });
  it('preserves the complete source pipeline without microbatches and its pinned Apache evidence', () => {
    const [row] = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/pipeline-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    validateCuratedSourceAsset(row);
    for (const change of [{ sha256: '0'.repeat(64) }, { license: 'CC0' },
      { crop_box_px: [0, 0, 10, 10] }, { commit: '0'.repeat(40) },
      { source_image_url: row.source_image_url!.replace(row.commit!, 'main') }])
      expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
  });
  it('retains the complete Ulysses figure and full pinned Ring GIF with separate rights evidence', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/tensor-sequence-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(2);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const change of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { license: 'CC0' }, { crop_box_px: [0, 0, 10, 10] }, { source_image_url: 'https://example.org/image.png' }])
        expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
    }
  });
  it('retains complete PyTorch checkpointing originals and pinned BSD evidence', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/checkpoint-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(2);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const change of [{ commit: '0'.repeat(40) }, { sha256: '0'.repeat(64) },
        { license: 'CC0' }, { crop_box_px: [0, 0, 10, 10] },
        { source_url: row.source_url!.replace('/blob/' + row.commit, '/blob/main') }])
        expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
      expect(readFileSync(resolve(root, 'site/public/licenses/pytorch-site-BSD-3.txt'))).toEqual(
        readFileSync(resolve(root, row.license_evidence!)));
    }
  });
  it('retains the complete EDLS error-feedback slide with pinned course MIT evidence', () => {
    const [row] = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/ddp-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    validateCuratedSourceAsset(row);
    for (const change of [{ source_page: 59 }, { sha256: '0'.repeat(64) },
      { source_asset_sha256: '0'.repeat(64) }, { license: 'CC0' },
      { crop_box_px: [0, 0, 100, 100] }, { render_page_size_px: [1600, 900] }])
      expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
    expect(readFileSync(resolve(root, 'site/public/licenses/edls-e632aa89-MIT.txt'))).toEqual(
      readFileSync(resolve(root, row.license_evidence!)));
  });
  it('retains two complete pinned HF DDP timelines and their Apache license evidence', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/distributed-overview-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(2);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const change of [{ commit: '0'.repeat(40) }, { sha256: '0'.repeat(64) },
        { source_asset_sha256: '0'.repeat(64) }, { license: 'CC0' },
        { source_url: row.source_url!.replace('/blob/' + row.commit, '/blob/main') },
        { crop_box_px: [0, 0, 10, 10] }])
        expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
    }
    expect(readFileSync(resolve(root, 'site/public/licenses/ultrascale-Apache-2.0.txt'))).toEqual(
      readFileSync(resolve(root, '05 Источники/Originals/distributed-overview-2026-10-23/Apache-2.0.txt')));
  });
  it('keeps the original Porian SVG and two microsoft/mup coordinate plots with separate rights', () => {
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/scaling-2026-10-23/assets.json'), 'utf8')) as CuratedAsset[];
    expect(rows).toHaveLength(3);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      expect(() => validateCuratedSourceAsset({ ...row, sha256: '0'.repeat(64) })).toThrow();
      if (row.asset.endsWith('.svg')) {
        expect(row.license).toBe('CC BY 4.0');
        expect(readFileSync(resolve(root, row.license_evidence!), 'utf8')).toContain('License: CC BY 4.0');
      } else {
        expect(row.license).toBe('MIT');
        expect(readFileSync(resolve(root, row.license_evidence!), 'utf8')).toContain('Copyright (c) Microsoft Corporation');
        expect(pngDimensions(resolve(root, row.asset))).toEqual([1500, 400]);
      }
    }
  });
  it('retains three unchanged HF animations without confusing the dataset and documentation licenses', () => {
    const rows = (parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] })
      .assets.filter(row => row.provenance_confirmation === 'pinned-hf-dataset-animation');
    expect(rows).toHaveLength(3);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const change of [{ commit: '0'.repeat(40) }, { sha256: '0'.repeat(64) },
        { source_asset_sha256: '0'.repeat(64) }, { license: 'Apache-2.0' },
        { source_url: row.source_url!.replace('/blob/' + row.commit, '/blob/main') },
        { crop_box_px: [0, 0, 10, 10] }])
        expect(() => validateCuratedSourceAsset({ ...row, ...change })).toThrow();
    }
  });
  it('rejects changed bytes or false rights in the three retained MoE image records',()=>{
    const rows=(parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')) as {assets:CuratedAsset[]})
      .assets.filter(a=>a.source_asset?.startsWith('05 Источники/Originals/moe-2026-10-23/')&&a.source_asset?.endsWith('.png'));
    expect(rows).toHaveLength(3);
    for(const row of rows){
      validateCuratedSourceAsset(row);
      for(const mutation of [{source_asset:'raw/absent.png'},{sha256:'0'.repeat(64)},
        {source_asset_sha256:'0'.repeat(64)},{author:'Unknown'},{license:'CC0'}])
        expect(()=>validateCuratedSourceAsset({...row,...mutation})).toThrow();
    }
  });
  it('verifies eight complete MoE source figures and rejects changed provenance',()=>{
    const rows=(parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')) as {assets:CuratedAsset[]})
      .assets.filter(a=>a.asset.includes('/moe-2026-10-23/'));
    expect(rows).toHaveLength(8);
    for(const row of rows){
      validateCuratedSourceAsset(row);
      for(const mutation of [{source_url:'https://example.org/'},{sha256:'0'.repeat(64)},
        {source_asset_sha256:'0'.repeat(64)},{license_evidence_sha256:'0'.repeat(64)},
        {render_dpi:1},{license:'CC0'},{author:'Unknown'}])
        expect(()=>validateCuratedSourceAsset({...row,...mutation})).toThrow();
    }
  });
  it('verifies the complete original Geva composition plot and its actual v1 license',()=>{
    const rows=(parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')) as {assets:CuratedAsset[]}).assets;
    const row=rows.find(a=>a.asset==='00 Учебник/Assets/Figures/curated/ffn-2026-10-23/geva-composition.svg')!;
    validateCuratedSourceAsset(row);
    for(const mutation of [{source_url:'https://example.org/'},{sha256:'0'.repeat(64)},
      {source_asset_sha256:'0'.repeat(64)},{source_page_sha256:'0'.repeat(64)},
      {license:'MIT'},{author:'Unknown'},{license_evidence_sha256:'0'.repeat(64)}])
      expect(()=>validateCuratedSourceAsset({...row,...mutation})).toThrow();
  });
  it('verifies complete Mistral, CLA and RULER originals and rejects altered provenance',()=>{
    const rows=(parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')) as {assets:CuratedAsset[]})
      .assets.filter(a=>a.asset.includes('/long-context-2026-10-23/'));
    expect(rows).toHaveLength(3);
    for(const row of rows){
      validateCuratedSourceAsset(row);
      for(const mutation of [{source_url:'https://example.org/'},{sha256:'0'.repeat(64)},
        {source_asset_sha256:'0'.repeat(64)},{source_page:1},{crop_box_px:[0,0,1,1]},
        {license:'MIT'},{author:'Unknown'},{license_evidence_sha256:'0'.repeat(64)}])
        expect(()=>validateCuratedSourceAsset({...row,...mutation})).toThrow();
    }
  });
  it('checks the MLA originals and historical copies and rejects false rights or changed source metadata',()=>{
    const specs=JSON.parse(readFileSync(resolve(root,'publishing/tools/mla-figures.json'),'utf8'));
    const rows=(parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')) as {assets:CuratedAsset[]}).assets;
    expect(specs).toHaveLength(5);
    for(const spec of specs){
      const row=rows.find(r=>r.asset===spec.asset)!;
      validateCuratedSourceAsset(row);
      for(const mutation of [{source_url:'https://example.org/'},{sha256:'0'.repeat(64)},
        {source_asset_sha256:'0'.repeat(64)},{author:'Unknown'},
        {rights_status:'CC0'},{metadata_status:'invented'}])
        expect(()=>validateCuratedSourceAsset({...row,...mutation})).toThrow();
      if(spec.crop_box_px){
        for(const mutation of [{crop_box_px:[0,0,1,1]},{license:'CC0'},
          {license_evidence_sha256:'0'.repeat(64)},{source_page_sha256:'0'.repeat(64)}])
          expect(()=>validateCuratedSourceAsset({...row,...mutation})).toThrow();
      }else if(spec.rights_status==='paper-figure-license-unverified')
        expect(()=>validateCuratedSourceAsset({...row,license:'MIT'})).toThrow();
    }
    const originals=JSON.parse(readFileSync(resolve(root,'publishing/tools/deepseek-v2-mla-originals.json'),'utf8')) as CuratedAsset[];
    expect(originals.map(row=>row.source_page)).toEqual([7,32,32]);
    for(const original of originals){
      const row=rows.find(r=>r.asset===original.asset)!;
      validateCuratedSourceAsset(row);
      for(const mutation of [{license:'CC0'},{source_url:'https://example.org/'},
        {source_page:1},{crop_box_px:[0,0,1,1]},{sha256:'0'.repeat(64)},{author:'Unknown'}])
        expect(()=>validateCuratedSourceAsset({...row,...mutation})).toThrow();
    }
  });
  it('checks four complete GQA results against ACL source, geometry and actual reuse license',()=>{
    const rows=(parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')) as {assets:CuratedAsset[]})
      .assets.filter(a=>a.asset.includes('/gqa-2026-10-22/'));
    expect(rows).toHaveLength(4);
    for(const row of rows){
      validateCuratedSourceAsset(row);
      for(const mutation of [{source_url:'https://example.org/'},{sha256:'0'.repeat(64)},
        {source_asset_sha256:'0'.repeat(64)},{source_page:1},{crop_box_px:[0,0,1,1]},
        {license:'MIT'},{author:'Unknown'},{license_evidence_sha256:'0'.repeat(64)}])
        expect(()=>validateCuratedSourceAsset({...row,...mutation})).toThrow();
    }
  });
  it('verifies ten complete CS336 pages and the exact original JAX batch graph',()=>{
    const rows=(parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')) as {assets:CuratedAsset[]})
      .assets.filter(a=>a.asset.includes('/transformer-shapes-2026-10-22/'));
    expect(rows).toHaveLength(11);
    for(const row of rows){
      validateCuratedSourceAsset(row);
      for(const mutation of [{source_url:'https://example.org/'},{sha256:'0'.repeat(64)},
        {source_asset_sha256:'0'.repeat(64)},{license:'CC0'},{author:'Unknown'},
        {crop_box_px:[0,0,1,1]},{license_evidence_sha256:'0'.repeat(64)}])
        expect(()=>validateCuratedSourceAsset({...row,...mutation})).toThrow();
    }
  });
  it('preserves the complete RoFormer/PI figures and rejects provenance drift',()=>{
    const rows=(parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')) as {assets:CuratedAsset[]})
      .assets.filter(a=>a.asset.includes('/rope-2026-10-22/'));
    expect(rows).toHaveLength(2);
    for(const row of rows){
      validateCuratedSourceAsset(row);
      for(const mutation of [{sha256:'0'.repeat(64)},{source_asset_sha256:'0'.repeat(64)},
        {source_page:1},{crop_box_px:[0,0,1,1]},{license:'MIT'},
        {source_url:'https://example.org/'},{author:'Unknown'},
        {license_evidence_sha256:'0'.repeat(64)}])
        expect(()=>validateCuratedSourceAsset({...row,...mutation})).toThrow();
    }
  });
  it('checks retained n-gram figures against exact Voita originals and rejects invented licenses', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(a => a.asset.startsWith('00 Учебник/Assets/Figures/curated/voita-language-modeling/'));
    expect(entries).toHaveLength(4);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_page_sha256: '0'.repeat(64) }, { source_image_url: 'https://example.org/' },
        { source_url: 'https://example.org/' }, { license: 'MIT' },
        { rights_status: 'licensed' }, { metadata_status: 'verified' }]) {
        expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
      }
    }
  });
  it('matches retained Voita figures to originals without inventing a named license', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(a => /\/voita-word-embeddings\/(?:one-hot\.png|lookup-table\.gif)$/.test(a.asset));
    expect(entries).toHaveLength(2);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_page_sha256: '0'.repeat(64) }, { source_image_url: 'https://example.org/' },
        { license: 'MIT' }, { rights_status: 'licensed' }, { metadata_status: 'verified' }]) {
        expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
      }
    }
  });
  it('checks normalization source figures, licenses and exact source provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(a => a.provenance_confirmation === 'norm-source-snapshot');
    expect(entries).toHaveLength(3);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const mutation of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { license: 'unknown' }, { license_evidence_sha256: '0'.repeat(64) },
        { source_url: 'https://example.org/not-the-source' },
        { provenance_confirmation: 'user-confirmed-open-materials' }]) {
        expect(() => validateCuratedSourceAsset({ ...entry, ...mutation })).toThrow();
      }
    }
    const crop = entries.find(e => e.source_page === 13)!;
    for (const mutation of [{ source_page: 1 }, { render_dpi: 100 }, { crop_box_px: [0, 0, 1, 1] }]) {
      expect(() => validateCuratedSourceAsset({ ...crop, ...mutation })).toThrow();
    }
  });
  it('checks original optimization figures and rejects unsupported provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(a => a.provenance_confirmation === 'optimization-source-snapshot');
    expect(entries).toHaveLength(3);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const mutation of [{ sha256: '0'.repeat(64) }, { license: 'unknown' },
        { source_page_sha256: '0'.repeat(64) }, { source_image_url: 'https://example.org/not-the-source' },
        { provenance_confirmation: 'user-confirmed-open-materials' }]) {
        expect(() => validateCuratedSourceAsset({ ...entry, ...mutation })).toThrow();
      }
    }
  });
  it('verifies the three original CS231n inline backprop figures and rejects stale provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(a => a.provenance_confirmation === 'pinned-cs231n-inline-svg');
    expect(entries).toHaveLength(3);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const mutation of [{ commit: '0'.repeat(40) }, { license: 'unknown' },
        { source_asset_sha256: '0'.repeat(64) }, { sha256: '0'.repeat(64) },
        { provenance_confirmation: 'user-confirmed-open-materials' }]) {
        expect(() => validateCuratedSourceAsset({ ...entry, ...mutation })).toThrow();
      }
    }
  });
  it('verifies author/course activation figures, precise crops, licenses and immutable bytes', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    for (const name of ['gelu-figure-1.png', 'cs336-silu-vs-relu.png']) {
    const entry = registry.assets.find(a => a.asset.endsWith('/' + name))!;
    expect(entry).toBeDefined();
    validateCuratedSourceAsset(entry);
    for (const invalid of [
      { source_url: entry.source_url!.replace(entry.commit!, 'master') },
      { license: 'unknown' }, { source_page: 1 }, { render_dpi: 100 },
      { crop_box_px: [912, 1040, 1696, 1590] }, { sha256: '0'.repeat(64) },
      { source_asset_sha256: '0'.repeat(64) }, { license_evidence_sha256: '0'.repeat(64) },
      { provenance_confirmation: 'user-confirmed-open-materials' }
    ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
    }
  });
  it('preserves the complete original TACO examples and English caption', () => {
    const entries = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/coding-web-computer-use-2026-10/assets.json'), 'utf8')) as CuratedAsset[];
    expect(entries).toHaveLength(1);
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    expect(registry.assets.filter(row => row.asset.includes('/coding-web-computer-use-2026-10/'))).toEqual(entries);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { license: 'public domain' },
        { source_page: 99 }, { author: '' }, { crop_box_px: [0, 0, 1, 1] }])
        expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
    }
  });
  it('preserves complete AgentBench, SWE-bench, tau-bench and PaperBench originals', () => {
    const entries = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/agent-evaluation-2026-10/assets.json'), 'utf8')) as CuratedAsset[];
    expect(entries).toHaveLength(5);
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    expect(registry.assets.filter(row => row.asset.includes('/agent-evaluation-2026-10/'))).toEqual(entries);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { license: 'public domain' },
        { source_page: 99 }, { author: '' }, { crop_box_px: [0, 0, 1, 1] }])
        expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
    }
  });
  it('preserves complete original Tree of Thoughts and Reflexion figures', () => {
    const entries = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/agent-memory-planning-2026-10/assets.json'), 'utf8')) as CuratedAsset[];
    expect(entries).toHaveLength(2);
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    expect(registry.assets.filter(row => row.asset.includes('/agent-memory-planning-2026-10/'))).toEqual(entries);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { license: 'public domain' },
        { source_page: 99 }, { author: '' }, { crop_box_px: [0, 0, 1, 1] }])
        expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
    }
  });
  it('preserves licensed harness workflow and controlled long-context experiment figures', () => {
    const entries = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/agent-harness-2026-10/assets.json'), 'utf8')) as CuratedAsset[];
    expect(entries).toHaveLength(3);
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    expect(registry.assets.filter(row => row.asset.includes('/agent-harness-2026-10/'))).toEqual(entries);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { license: 'public domain' },
        { source_page: 99 }, { author: '' }, { crop_box_px: [0, 0, 1, 1] }])
        expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
    }
  });
  it('preserves both complete numbered ReAct panels with pinned CC BY evidence', () => {
    const entries = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/tool-use-2026-10/assets.json'), 'utf8')) as CuratedAsset[];
    expect(entries).toHaveLength(2);
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    expect(registry.assets.filter(row => row.asset.includes('/tool-use-2026-10/'))).toEqual(entries);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { license: 'public domain' },
        { source_page: 99 }, { author: '' }, { crop_box_px: [0, 0, 1, 1] }])
        expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
    }
  });
  it('preserves five licensed primary-paper Chameleon figures at readable source resolution', () => {
    const entries = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/chameleon-2026-10/assets.json'), 'utf8')) as CuratedAsset[];
    expect(entries).toHaveLength(5);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { license: 'public domain' },
        { source_page: 99 }, { author: '' }, { crop_box_px: [0, 0, 1, 1] }])
        expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
    }
  });
  it('preserves seven diffusion source objects, including the MIT no-derivatives license', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const figures = registry.assets.filter(row => row.asset.includes('/diffusion-flow-2026-10/'));
    expect(figures).toHaveLength(7);
    expect(figures.filter(row => row.license === 'CC BY-NC-ND 4.0')).toHaveLength(5);
    for (const row of figures) {
      validateCuratedSourceAsset(row);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { license: 'public-domain' },
        { source_page: 99 }, { author: '' }, { crop_box_px: [0, 0, 1, 1] },
        { source_url: 'https://example.org/' }])
        expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });
  it('verifies the retrieval documentation images and the licensed complete HyDE figure crop', () => {
    const entries = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/retrieval-review-2026-10/assets.json'), 'utf8')) as CuratedAsset[];
    expect(entries).toHaveLength(6);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [
        { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { license: 'public domain' }, { license_evidence_sha256: '0'.repeat(64) },
        { source_url: 'https://example.org/' }, { author: '' }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
      if (entry.source_page) {
        expect(() => validateCuratedSourceAsset({ ...entry, source_page: 1 })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...entry, crop_box_px: [200, 200, 1630, 580] })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...entry, source_page_sha256: '0'.repeat(64) })).toThrow();
      } else {
        expect(() => validateCuratedSourceAsset({ ...entry, commit: '0'.repeat(40) })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...entry, source_code_sha256: '0'.repeat(64) })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...entry, crop_box_px: [0, 0, 100, 100] })).toThrow();
      }
    }
  });
  it('rejects mismatched reranking source pages, crops and license identities', () => {
    const entries = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/reranking-review-2026-10/assets.json'), 'utf8')) as CuratedAsset[];
    expect(entries).toHaveLength(7);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const field of ['sha256', 'source_asset_sha256', 'license_evidence_sha256'])
        expect(() => validateCuratedSourceAsset({ ...entry, [field]: '0'.repeat(64) })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, author: '' })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, source_url: 'https://example.org/unrelated' })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, license: 'unverified' })).toThrow();
      if (entry.source_page) {
        expect(() => validateCuratedSourceAsset({ ...entry, source_page: 99 })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...entry, crop_box_px: [0, 0, 1, 1] })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...entry, source_page_sha256: '0'.repeat(64) })).toThrow();
      } else {
        expect(() => validateCuratedSourceAsset({ ...entry, commit: '0'.repeat(40) })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...entry, source_code_sha256: '0'.repeat(64) })).toThrow();
      }
    }
  });
  it('retains five unchanged embedding illustrations with separate upstream licenses', () => {
    const entries = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/embeddings-review-2026-10/assets.json'), 'utf8')) as CuratedAsset[];
    expect(entries).toHaveLength(5);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [
        { commit: '0'.repeat(40) }, { source_image_url: entry.source_image_url?.replace(entry.commit!, 'main') },
        { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { license: 'public domain' }, { license_evidence_sha256: '0'.repeat(64) },
        { source_page_sha256: '0'.repeat(64) }, { derivation: 'source-raster-crop' },
        { crop_box_px: [0, 0, 10, 10] }, { author: '' }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
    }
  });
  it('keeps SRE no-derivatives figures unchanged and retains the license evidence', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(a => a.provenance_confirmation === 'official-sre-book-asset');
    expect(entries).toHaveLength(2);
    for (const entry of entries) {
      expect(() => validateCuratedSourceAsset(entry)).not.toThrow();
      for (const changed of [
        { source_asset_sha256: '0'.repeat(64) },
        { license_evidence_sha256: '0'.repeat(64) },
        { license: 'CC BY-NC-SA 4.0' },
        { crop_box_px: [0, 0, 10, 10] },
        { derivation: 'source-raster-crop' },
        { source_image_url: entry.source_image_url!.replace('sre.google', 'example.org') }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...changed })).toThrow();
    }
  });
  it('decodes rendered PNG pixel streams and rejects corrupt or incomplete image data', () => {
    function chunk(kind: string, data: Buffer): Buffer {
      const bytes = Buffer.alloc(data.length + 12);
      bytes.writeUInt32BE(data.length, 0);
      bytes.write(kind, 4, 'ascii');
      data.copy(bytes, 8);
      bytes.writeUInt32BE(pngCrc(bytes.subarray(4, -4)), bytes.length - 4);
      return bytes;
    }
    function png(pixels = Buffer.from([0, 255, 0, 0, 0, 255, 0]), compressed?: Buffer): Buffer {
      const header = Buffer.from([0, 0, 0, 2, 0, 0, 0, 1, 8, 2, 0, 0, 0]);
      return Buffer.concat([
        Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
        chunk('IHDR', header), chunk('IDAT', compressed ?? deflateSync(pixels)), chunk('IEND', Buffer.alloc(0))
      ]);
    }
    const valid = png();
    expect(validateRenderedPngBytes(valid)).toEqual([2, 1]);
    const badCrc = Buffer.from(valid);
    badCrc[32] ^= 1;
    for (const invalid of [
      Buffer.from('<html>IEND</html>'),
      valid.subarray(0, -12),
      Buffer.concat([valid, Buffer.from('extra')]),
      badCrc,
      png(undefined, Buffer.from('not a zlib pixel stream')),
      png(Buffer.from([0, 255, 0])),
      png(Buffer.from([5, 255, 0, 0, 0, 255, 0]))
    ]) expect(() => validateRenderedPngBytes(invalid)).toThrow();
  });

  it('verifies complete original SpanBERT crops and rejects changed page, pixels or rights', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as {assets: CuratedAsset[]};
    const rows = JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/bert-objectives-2026-10-29/assets.json'), 'utf8')) as CuratedAsset[];
    for (const spec of rows) {
      const entry = registry.assets.find(row => row.asset === spec.asset)!;
      expect(() => validateCuratedSourceAsset(entry)).not.toThrow();
      for (const invalid of [{license:'MIT'}, {source_page:9}, {render_dpi:144},
        {sha256:'0'.repeat(64)}, {source_asset_sha256:'0'.repeat(64)},
        {crop_box_px:[0,0,20,20]}, {asset:entry.asset.replace('.png','-changed.png')}])
        expect(() => validateCuratedSourceAsset({...entry,...invalid})).toThrow();
    }
  });

  it('pins the complete bidirectional SVG and rejects changed provenance',()=>{
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')) as {assets:CuratedAsset[]};
    const entry=registry.assets.find(x=>x.asset==='00 Учебник/Assets/Figures/curated/lstm-gru-contracts-2026-11-03/d2l-birnn.svg')!;
    expect(()=>validateCuratedSourceAsset(entry)).not.toThrow();
    for(const invalid of [{author:'Unknown'},{commit:'main'},{license:'CC0'},
      {sha256:'0'.repeat(64)},{source_asset_sha256:'0'.repeat(64)},
      {source_page_sha256:'0'.repeat(64)},{license_evidence_sha256:'0'.repeat(64)},
      {source_image_url:'https://example.org/a.svg'},{derivation:'source-raster-copy'}])
      expect(()=>validateCuratedSourceAsset({...entry,...invalid})).toThrow();
  });

  it('verifies LSTM/GRU source figures and separates source proof from unconfirmed rights', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    for (const asset of Object.keys(lstmGruFigures)) {
      const entry = registry.assets.find(row => row.asset === asset)!;
      expect(entry).toBeDefined();
      expect(() => validateCuratedSourceAsset(entry)).not.toThrow();
      for (const invalid of [
        { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_asset: undefined }, { source_page_snapshot: undefined },
        { source_image_url: 'https://example.org/wrong.png' },
        { source_page_sha256: '0'.repeat(64) }, { provenance_confirmation: undefined },
        { license: 'CC0' }, { author: 'Unknown' },
        { metadata_status: entry.metadata_status === 'verified' ? 'needs-confirmation' : 'verified' },
        { derivation: 'pdf-page-crop' }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
      if (entry.metadata_status === 'verified') expect(() => validateCuratedSourceAsset({ ...entry, commit: 'main' })).toThrow();
      else expect(() => validateCuratedSourceAsset({ ...entry, rights_status: 'fully-confirmed' })).toThrow();
    }
  });

  it('verifies the five original RNN/BPTT SVGs and rejects changed source or license records', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    for (const asset of Object.keys(rnnBpttFigures)) {
      const entry = registry.assets.find(row => row.asset === asset)!;
      expect(entry).toBeDefined();
      expect(() => validateCuratedSourceAsset(entry)).not.toThrow();
      for (const invalid of [
        { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_asset: undefined }, { source_page_snapshot: undefined },
        { source_image_url: 'https://example.org/wrong.svg' }, { commit: 'main' },
        { source_page_sha256: '0'.repeat(64) }, { provenance_confirmation: undefined },
        { license: 'CC0' }, { license_url: undefined }, { author: 'Unknown' },
        { metadata_status: 'needs-confirmation' }, { derivation: 'pdf-page-crop' },
        { license_evidence_sha256: '0'.repeat(64) }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
    }
  });

  it('verifies word-embedding source figures, dimensions, licenses and unchanged bytes', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    for (const asset of Object.keys(wordEmbeddingFigures)) {
      const entry = registry.assets.find(row => row.asset === asset)!;
      expect(entry).toBeDefined();
      expect(() => validateCuratedSourceAsset(entry)).not.toThrow();
      for (const invalid of [
        { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_asset: undefined }, { source_page_snapshot: undefined },
        { source_image_url: 'https://example.org/wrong.png' },
        { source_page_sha256: '0'.repeat(64) }, { provenance_confirmation: undefined },
        { license: 'CC0' }, { author: 'Unknown' },
        { metadata_status: entry.metadata_status === 'verified' ? 'needs-confirmation' : 'verified' },
        { derivation: 'pdf-page-crop' }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
    }
  });

  it('verifies the complete neural-LM source SVG and rejects altered provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'));
    const valid = registry.assets.find((entry: CuratedAsset) => entry.id === 'curated-neural-lm-d2l-softmaxreg-f426511da025');
    expect(valid).toBeDefined();
    expect(() => validateCuratedSourceAsset(valid)).not.toThrow();
    expect(valid.used_in).toContain('00 Учебник/00 Математические и ML-основания/04 Функции потерь.md');
    for (const invalid of [
      { source_image_url: valid.source_image_url.replace(d2lCommit, 'main') },
      { commit: '0'.repeat(40) }, { sha256: '0'.repeat(64) },
      { source_asset_sha256: '0'.repeat(64) }, { source_page_sha256: '0'.repeat(64) },
      { source_asset: undefined }, { source_page_snapshot: undefined },
      { license: 'CC BY 4.0' }, { license_url: undefined },
      { license_evidence_sha256: '0'.repeat(64) }, { license_evidence: undefined },
      { metadata_status: 'needs-confirmation' }, { provenance_confirmation: undefined },
      { derivation: 'source-raster-copy' }
    ]) expect(() => validateCuratedSourceAsset({ ...valid, ...invalid })).toThrow();
  });

  it.each([
    ['d2l-language-model-data', 'lang-model-data', 'chapter_recurrent-neural-networks/language-model.html#fig-lang-model-data'],
    ['d2l-gpt-decoder', 'gpt-decoder-only', 'chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html#gpt-and-gpt-2']
  ])('verifies a full D2L SVG rasterization and its pinned source: %s', (localName, upstreamName, page) => {
    const directory = '00 Учебник/Assets/Figures/curated/ntp-review-2026-10';
    const source_asset = `${directory}/${localName}-original.svg`;
    const asset = `${directory}/${localName}.png`;
    const valid: CuratedAsset = {
      asset, source_asset, author: 'Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola',
      source_url: `https://d2l.ai/${page}`,
      source_image_url: `https://raw.githubusercontent.com/d2l-ai/d2l-en/${d2lCommit}/img/${upstreamName}.svg`,
      commit: d2lCommit, license: 'CC BY-SA 4.0',
      license_url: `https://github.com/d2l-ai/d2l-en/blob/${d2lCommit}/LICENSE`,
      source_asset_sha256: sha256(resolve(root, source_asset)), sha256: sha256(resolve(root, asset)),
      derivation: 'source-svg-rasterization', provenance_confirmation: 'source-license-verified',
      metadata_status: 'verified', modifications: 'Full SVG rasterized to PNG without a crop.'
    };
    expect(() => validateCuratedSourceAsset(valid)).not.toThrow();
    const otherSource = `${directory}/${localName === 'd2l-gpt-decoder' ? 'd2l-language-model-data' : 'd2l-gpt-decoder'}-original.svg`;
    expect(() => validateCuratedSourceAsset({
      ...valid, source_asset: otherSource, source_asset_sha256: sha256(resolve(root, otherSource))
    })).toThrow(); // Both files are valid, but the output is not a full render of this viewBox.
    for (const invalid of [
      { source_asset: undefined },
      { source_asset: source_asset.replace('/curated/', '/curated/../curated/') },
      { source_asset: source_asset.replace('-original.svg', '.png') },
      { source_asset: source_asset.replace('-original.svg', '-missing-original.svg') },
      { asset: source_asset },
      { asset: asset.replace('/curated/', '/curated/../curated/') },
      { asset: asset.replace('.png', '.jpg') },
      { source_url: undefined },
      { source_url: valid.source_url?.replace('d2l.ai', 'example.org') },
      { source_image_url: undefined },
      { source_image_url: valid.source_image_url?.replace(d2lCommit, 'main') },
      { source_image_url: valid.source_image_url?.replace('d2l-en', 'unrelated') },
      { source_image_url: valid.source_image_url?.replace('.svg', '.png') },
      { source_image_url: valid.source_image_url?.replace('/img/', '/img/../img/') },
      { commit: undefined },
      { commit: '0'.repeat(40) },
      { source_asset_sha256: undefined },
      { source_asset_sha256: '0'.repeat(64) },
      { sha256: undefined },
      { sha256: '0'.repeat(64) },
      { author: '' },
      { license: 'CC BY 4.0' },
      { license_url: undefined },
      { license_url: valid.license_url?.replace(d2lCommit, 'main') },
      { metadata_status: 'needs-confirmation' },
      { provenance_confirmation: 'user-confirmed-open-materials' },
      { derivation: 'source-raster-copy' },
      { derivation: 'source-svg-copy' },
      { modifications: '' },
      { crop_box_px: [0, 0, 2000, 292] },
      { render_page_size_px: [2000, 292] }
    ]) expect(() => validateCuratedSourceAsset({ ...valid, ...invalid })).toThrow();
  });

  it('verifies existing Stanford PDF chart crops without upgrading unconfirmed third-party rights', () => {
    const source_asset = '05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/lecture_09.pdf';
    const asset = '00 Учебник/Assets/Figures/curated/scaling-review-2026-10/data-power-law-chart.png';
    const valid: CuratedAsset = {
      asset, source_asset, source_page: 15,
      source_asset_sha256: sha256(resolve(root, source_asset)), sha256: sha256(resolve(root, asset)),
      source_url: `https://github.com/stanford-cs336/lectures/blob/${cs336Commit}/lecture_09.pdf#page=15`,
      commit: cs336Commit, author: 'Stanford CS336 course staff; underlying chart authors retained in NOTICE',
      derivation: 'pdf-page-render-crop', provenance_confirmation: 'user-confirmed-open-materials',
      metadata_status: 'needs-confirmation', license: 'User permission record (not a named license)',
      license_url: 'https://cs336.stanford.edu/',
      rights_status: 'existing-local-reformat; third-party-rights-unverified',
      modifications: 'Crop of the existing local PDF render; no redraw.',
      render_page_size_px: [2400, 1352], crop_box_px: [636, 473, 1524, 1169]
    };
    expect(() => validateCuratedSourceAsset(valid)).not.toThrow();
    for (const invalid of [
      { source_asset: undefined },
      { source_asset: 'raw/papers/external/course.pdf' },
      { source_asset: source_asset.replace('/repository/', '/repository/../repository/') },
      { source_asset: source_asset.replace('lecture_09.pdf', 'lecture_10.pdf') },
      { source_asset: source_asset.replace('lecture_09.pdf', 'lecture_11.pdf') },
      { source_asset: asset },
      { source_asset_sha256: '0'.repeat(64) },
      { sha256: '0'.repeat(64) },
      { source_url: undefined },
      { source_url: valid.source_url?.replace(cs336Commit, 'main') },
      { source_url: valid.source_url?.replace('#page=15', '#page=14') },
      { source_url: valid.source_url?.replace('lecture_09.pdf', 'lecture_11.pdf') },
      { source_url: valid.source_url?.replace('stanford-cs336', 'another-owner') },
      { commit: '0'.repeat(40) },
      { commit: undefined },
      { source_page: 0 },
      { source_page: 15.5 },
      { source_page: undefined },
      { source_page: 58, source_url: valid.source_url?.replace('#page=15', '#page=58') },
      { asset: asset.replace('/curated/', '/curated/../curated/') },
      { asset: asset.replace('.png', '.jpg') },
      { author: '' },
      { metadata_status: 'verified' },
      { license: 'CC BY 4.0' },
      { license_url: undefined },
      { rights_status: 'existing-local-reformat' },
      { rights_status: undefined },
      { provenance_confirmation: 'source-license-verified' },
      { derivation: 'source-raster-copy' },
      { modifications: '' },
      { render_page_size_px: undefined },
      { render_page_size_px: [2400, 1350] },
      { crop_box_px: undefined },
      { crop_box_px: '636,473,1524,1169' },
      { crop_box_px: [636, 473, 1524] },
      { crop_box_px: [636, 473, 1524, 1169, 0] },
      { crop_box_px: [636.5, 473, 1524, 1169] },
      { crop_box_px: [-1, 473, 887, 1169] },
      { crop_box_px: [636, -1, 1524, 695] },
      { crop_box_px: [636, 473, 636, 1169] },
      { crop_box_px: [636, 473, 1524, 473] },
      { crop_box_px: [1524, 473, 636, 1169] },
      { crop_box_px: [636, 1169, 1524, 473] },
      { crop_box_px: [1513, 473, 2401, 1169] },
      { crop_box_px: [636, 657, 1524, 1353] },
      { crop_box_px: [636, 473, 1525, 1169] }
    ]) expect(() => validateCuratedSourceAsset({ ...valid, ...invalid })).toThrow();
  });

  it('verifies pinned Hugging Face Space SVG and PNG copies and rejects mismatched provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const fixtures = [
      'arithmetic-review-2026-09/hf-training-memory.svg',
      'arithmetic-review-2026-09/hf-fsdp-forward.svg',
      'arithmetic-review-2026-09/hf-fsdp-backward.svg',
      'arithmetic-review-2026-09/hf-pipeline-gpipe.svg',
      'arithmetic-review-2026-09/hf-pipeline-1f1b.svg',
      'precision-review-2026-09/hf-format-ranges.png',
      'precision-review-2026-09/hf-format-resolution.png'
    ];
    const entries = fixtures.map(filename => registry.assets.find(asset =>
      asset.asset === `00 Учебник/Assets/Figures/curated/${filename}`
    )!);
    for (const entry of entries) {
      expect(entry).toBeTruthy();
      expect(() => validateCuratedSourceAsset(entry)).not.toThrow();
      const wrongExtension = entry.asset.endsWith('.svg') ? '.png' : '.svg';
      for (const invalid of [
        { source_url: undefined },
        { source_url: entry.source_url?.replace(`/blob/${entry.commit}/`, '/blob/main/') },
        { source_image_url: entry.source_image_url?.replace(`/resolve/${entry.commit}/`, '/resolve/main/') },
        { source_url: entry.source_url?.replace(entry.commit!, '0'.repeat(40)) },
        { source_image_url: entry.source_image_url?.replace(entry.commit!, '0'.repeat(40)) },
        { commit: '0'.repeat(40) },
        { commit: undefined },
        { source_url: entry.source_url?.replace('/nanotron/', '/another-owner/') },
        { source_image_url: entry.source_image_url?.replace('/ultrascale-playbook/', '/another-repository/') },
        { source_url: entry.source_url?.replace(/\/[^/]+\.(svg|png)$/, `/another-figure${extname(entry.asset)}`) },
        { source_image_url: entry.source_image_url?.replace(/\/[^/]+\.(svg|png)$/, `/another-figure${extname(entry.asset)}`) },
        { source_image_url: entry.source_image_url?.replace(/\.(svg|png)$/, wrongExtension) },
        {
          source_url: entry.source_url?.replace(/\.(svg|png)$/, wrongExtension),
          source_image_url: entry.source_image_url?.replace(/\.(svg|png)$/, wrongExtension)
        },
        { source_image_url: undefined },
        { asset: entry.asset.replace('/curated/', '/curated/../curated/') },
        { asset: entry.asset.replace(/\.(svg|png)$/, wrongExtension) },
        { sha256: '0'.repeat(64) },
        { sha256: undefined },
        { provenance_confirmation: undefined },
        { author: '' },
        { license: '' },
        { license_url: undefined },
        { modifications: '' },
        { metadata_status: 'unverified' },
        { derivation: entry.derivation === 'source-svg-copy' ? 'source-raster-copy' : 'source-svg-copy' },
        { derivation: undefined }
      ]) {
        expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
      }
    }
  });

  it('verifies unchanged MIT OCW course images and rejects unrelated or incomplete provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(asset => asset.provenance_confirmation === 'official-course-asset');
    expect(entries).toHaveLength(2);
    for (const entry of entries) {
      expect(() => validateCuratedSourceAsset(entry)).not.toThrow();
      for (const invalid of [
        { source_image_url: 'https://example.org/figure.png' },
        { source_image_url: entry.source_image_url?.replace('6-004-computation-structures-spring-2017', 'another-course') },
        { source_image_url: entry.source_image_url?.replace(/\.png$/, '.html') },
        { source_url: undefined },
        { author: '' },
        { license: 'unknown' },
        { license_url: undefined },
        { sha256: '0'.repeat(64) },
        { source_asset_sha256: undefined },
        { metadata_status: 'unverified' },
        { derivation: 'redraw' },
        { modifications: '' }
      ]) {
        expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
      }
    }
  });

  it.each([
    [0, [375, 0, 900, 322]],
    [1, [945, 0, 1475, 322]],
    [2, [1520, 0, 2039, 322]]
  ])('verifies licensed tracked PNG crop panel %s and rejects invalid provenance or bounds', (step, crop_box_px) => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entry = registry.assets.find(asset => asset.asset.endsWith(`/mistral-rolling-step-${step}.png`))!;
    expect(entry).toBeTruthy();
    const valid: CuratedAsset = {
      ...entry,
      derivation: 'source-raster-crop',
      source_asset: '00 Учебник/Assets/Figures/curated/long-context-review-2026-09/mistral-rolling-buffer.png',
      crop_box_px
    };
    expect(() => validateCuratedSourceAsset(valid)).not.toThrow();
    for (const invalid of [
      { source_asset: undefined },
      { source_asset: '../unrelated-source.png' },
      { source_asset: valid.source_asset?.replace('/curated/', '/curated/../curated/') },
      { source_asset: valid.source_asset?.replace(/\.png$/, '.jpg') },
      { source_asset_sha256: undefined },
      { source_asset_sha256: '0'.repeat(64) },
      { sha256: undefined },
      { sha256: '0'.repeat(64) },
      { asset: valid.source_asset!, sha256: valid.source_asset_sha256 },
      { provenance_confirmation: 'user-confirmed-open-materials' },
      { author: '' },
      { license: '' },
      { license_url: undefined },
      { metadata_status: 'unverified' },
      { modifications: '' },
      { crop_box_px: undefined },
      { crop_box_px: '375,0,900,322' },
      { crop_box_px: [375, 0, 900] },
      { crop_box_px: [375, 0, 900, 322, 0] },
      { crop_box_px: [375.5, 0, 900, 322] },
      { crop_box_px: [375, -1, 900, 322] },
      { crop_box_px: [-1, 0, 900, 322] },
      { crop_box_px: [900, 0, 375, 322] },
      { crop_box_px: [375, 322, 900, 0] },
      { crop_box_px: [375, 0, 375, 322] },
      { crop_box_px: [375, 0, 900, 0] },
      { crop_box_px: [375, 0, 2040, 322] },
      { crop_box_px: [375, 0, 900, 323] },
      { crop_box_px: [375, 0, 901, 322] }
    ]) {
      expect(() => validateCuratedSourceAsset({ ...valid, ...invalid })).toThrow();
    }
  });

  it('verifies tracked paper crops and a version-pinned unchanged arXiv figure', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const crops = registry.assets.filter(asset => asset.source_asset?.startsWith('05 Источники/Papers/Source PDFs/'));
    expect(crops.length).toBeGreaterThanOrEqual(4);
    for (const entry of crops) {
      validateCuratedSourceAsset(entry);
      expect(() => validateCuratedSourceAsset({ ...entry, source_asset: '../paper.pdf' })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, source_asset_sha256: '0'.repeat(64) })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, sha256: '0'.repeat(64) })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, source_page: 0 })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, license: '' })).toThrow();
    }
    const original = registry.assets.find(asset => asset.source_image_url?.includes('2411.06559v1'))!;
    expect(original).toBeTruthy();
    validateCuratedSourceAsset(original);
    expect(() => validateCuratedSourceAsset({ ...original, source_url: 'https://arxiv.org/abs/2411.06559' })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...original, source_image_url: original.source_image_url?.replace('v1', 'v2') })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...original, sha256: '0'.repeat(64) })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...original, source_asset_sha256: undefined })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...original, license_url: undefined })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...original, derivation: 'redrawn' })).toThrow();
  });
  it('verifies licensed assignment figures and unchanged lecture-cache originals', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const ids = ['curated-stanford-a1-complete-model-block-e76f4539f68d', 'curated-jax-scaling-roofline-2c9018ab614d'];
    for (const id of ids) {
      const entry = registry.assets.find(asset => asset.id === id)!;
      expect(entry).toBeTruthy();
      validateCuratedSourceAsset(entry);
      expectDecodableImage(resolve(root, entry.asset));
      expect(() => validateCuratedSourceAsset({ ...entry, source_asset: '../unrelated-source' })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, source_asset_sha256: '0'.repeat(64) })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, sha256: '0'.repeat(64) })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, license_url: undefined })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, derivation: 'redrawn' })).toThrow();
      if (entry.derivation === 'pdf-page-render-crop') {
        expect(() => validateCuratedSourceAsset({ ...entry, source_page: 0 })).toThrow();
      }
    }
  });

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

  it('pins the three Stanford crops to Chen v1 and keeps the duplicate stats crop retired', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const ids = ['curated-spec-alg-000000000047', 'curated-spec-results-000000000048', 'curated-spec-stats-000000000049'];
    for (const id of ids) {
      const entry = registry.assets.find((asset) => asset.id === id);
      expect(entry).toBeTruthy();
      if (!entry) continue;
      expect(entry.author).toBe('Charlie Chen et al.');
      expect(entry.source_url).toBe('https://arxiv.org/abs/2302.01318v1');
      expect(entry.license).toBe('CC BY 4.0');
      expect(entry.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
      validateCuratedSourceAsset(entry);
      expect(() => validateCuratedSourceAsset({ ...entry, license: '' })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, license_url: undefined })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, source_page: 1 })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, commit: '0'.repeat(40) })).toThrow();
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
    // Imported course images have their own namespace and public registry checks.
    // Compare the complete textbook figure directory to its own registry rows.
    const textbookAssets = registry.assets.filter(asset =>
      typeof asset.asset === 'string' && asset.asset.startsWith('00 Учебник/Assets/Figures/'));
    expect(textbookAssets.map(asset => asset.asset).toSorted()).toEqual(files);
    for (const path of paths) expect(path).toMatch(/^(?:00 Учебник\/Assets\/Figures|Assets\/Sources)\//);
    expect(paths).toContain('00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png');
    for (const asset of textbookAssets) {
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
        if (vllmSourceLibrary.retired.some(row => row.asset === asset.asset) ||
            optimizerCurvatureAssets.some(row => row.asset === asset.asset) ||
            [...sequenceRecallReview.figures, ...sequenceRecallReview.retired].some(row => row.asset === asset.asset)) {
          validateCuratedSourceAsset(asset);
          continue;
        }
        const pendingSequence = sequenceFigureReview.figures.find(row =>
          row.asset === asset.asset && row.rights_status === 'paper-figure-license-unverified');
        if (pendingSequence) {
          expect(asset).toMatchObject({
            license: null, license_url: null, rights_status: 'paper-figure-license-unverified',
            provenance_confirmation: 'original-pdf-pixels-verified',
          });
          continue;
        }
        if (lmcacheTransferReview.figures.some(row => row.asset === asset.asset) ||
            lmcacheTransferReview.retired.some(row => row.asset === asset.asset) ||
            asset.asset === inferenceTimelineReview.figure.asset ||
            reconciledReaderFigures.some(row => row.asset === asset.asset)) {
          validateCuratedSourceAsset(asset);
          continue;
        }
        expect(typeof asset.license).toBe('string');
        if (asset.license_url === null) {
          // Verified source bytes do not imply a verified redistribution license.
          if (asset.publication_status === 'retired') {
            expect(asset.used_in).toEqual([]);
            expect(asset.retirement_reason).toEqual(expect.any(String));
            expect(asset.license).toBe('Unverified for redistribution');
            expect(asset.rights_status).toBe(asset.id === 'curated-cs229-bias-variance-0000000000ac'
              ? 'course-note-republication-unverified' : 'course-slide-republication-unverified');
            const archivedLecture2 = asset.id === 'curated-cs231n-softmax-ce-0000000000a6'
              || asset.id === 'curated-cs231n-boundaries-0000000000a7';
            expect(asset.provenance_confirmation).toBe(archivedLecture2
              ? 'source-pixels-verified' : 'source-linked-rights-unverified');
          } else {
            expect(asset.id).toMatch(/^curated-cs336-arc[123]-/);
            expect(asset.rights_status).toMatch(/^limited-educational-quotation/);
            expect(asset.provenance_confirmation).toBe('cs336-source-snapshot; figure-discussed-in-text');
          }
        } else expect(typeof asset.license_url).toBe('string');
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
    expect(deepseekR1).toMatchObject({ used_in: [], publication_status: 'retired' });
    expect(registry.assets.find(asset => asset.asset.endsWith('/r1-zero-nature-figure1.png'))?.used_in)
      .toContain('Papers/DeepSeek-R1 Reasoning via RL.md');
    const deepseekPipeline = registry.assets.find((asset) =>
      asset.asset === '00 Учебник/Assets/Figures/curated/deepseek-r1/figure2-pipeline.png'
    );
    expect(deepseekPipeline?.used_in).toContain(
      '00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1.md'
    );
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

  it('verifies MIT queue diagrams and complete unchanged SRE originals', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(entry => entry.asset.startsWith('00 Учебник/Assets/Figures/curated/inference-queueing-review-2026-10/'));
    expect(entries).toHaveLength(5);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [
        { author: '' },
        { license: '' },
        { license_url: 'https://example.org/license' },
        { source_url: 'https://example.org/source' },
        { sha256: '0'.repeat(64) },
        { source_asset_sha256: '0'.repeat(64) },
        { license_evidence_sha256: '0'.repeat(64) },
        { modifications: '' },
        { metadata_status: 'unverified' }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
      if (entry.asset.endsWith('.webp')) {
        expect(() => validateCuratedSourceAsset({ ...entry, crop_box_px: [0, 0, 100, 100] })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...entry, source_image_url: 'https://example.org/image.webp' })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...entry, derivation: 'source-raster-crop' })).toThrow();
      } else {
        expect(() => validateCuratedSourceAsset({ ...entry, source_page: 0 })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...entry, crop_box_px: [0, 0, 100, 100] })).toThrow();
      }
    }
  });

  it('pins the Gender Shades PDF crop and its publication grant', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entry = registry.assets.find(asset => asset.id === 'curated-gender-shades-figure3-3c0aed571f22')!;
    expect(entry).toBeDefined();
    expect(() => validateCuratedSourceAsset(entry)).not.toThrow();
    for (const invalid of [{ source_page: 9 }, { render_dpi: 72 }, { license: 'unverified' },
      { crop_box_px: [1544, 445, 2621, 826] }, { source_asset_sha256: '0'.repeat(64) }])
      expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
  });

  it('checks original SHAP and AIR-Bench figures and rejects provenance drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(entry => entry.asset.startsWith('00 Учебник/Assets/Figures/curated/responsible-review-2026-10/'));
    expect(entries).toHaveLength(3);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [
        { author: '' }, { modifications: '' }, { license: '' },
        { source_url: 'https://example.org/page' }, { source_image_url: 'https://example.org/image' },
        { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_page_sha256: '0'.repeat(64) }, { license_evidence_sha256: '0'.repeat(64) },
        { derivation: 'source-raster-crop' }, { crop_box_px: [0, 0, 100, 100] },
        { metadata_status: 'unverified' }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
      if (entry.asset.endsWith('.png')) {
        expect(() => validateCuratedSourceAsset({ ...entry, commit: '0'.repeat(40) })).toThrow();
      }
    }
  });

  it('checks ten imbalance gallery originals and exact source, version and license records', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(entry => entry.asset.startsWith('00 Учебник/Assets/Figures/curated/imbalance-review-2026-10/'));
    expect(entries).toHaveLength(10);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [
        { author: '' }, { modifications: '' }, { license: 'public access' },
        { source_url: 'https://example.org/source' }, { source_image_url: 'https://example.org/img.png' },
        { source_page_snapshot: '' }, { source_code: '' }, { commit: '0'.repeat(40) },
        { metadata_status: 'unverified' }, { derivation: 'redrawn' }, { crop_box_px: [0, 0, 100, 100] },
        { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_code_sha256: '0'.repeat(64) }, { source_page_sha256: '0'.repeat(64) },
        { license_evidence_sha256: '0'.repeat(64) }, { source_page: 1 }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
    }
  });

  it('checks five evaluation originals, animation identity and paper crop provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(entry => entry.asset.startsWith('00 Учебник/Assets/Figures/curated/evaluation-review-2026-10/'));
    expect(entries).toHaveLength(5);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [
        { author: '' }, { modifications: '' }, { license: 'public access' },
        { source_url: 'https://example.org/source' }, { metadata_status: 'unverified' },
        { license_url: 'https://example.org/license' },
        { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { license_evidence_sha256: '0'.repeat(64) }, { derivation: 'redrawn' },
        { crop_box_px: [0, 0, 100, 100] }, { source_page: 0 }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
      if (entry.asset.endsWith('.gif')) {
        expect(() => validateCuratedSourceAsset({ ...entry, commit: '0'.repeat(40) })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...entry, source_image_url: 'https://example.org/a.gif' })).toThrow();
      }
    }
  });

  it('checks six Edge originals, exact paper crops and license evidence', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(entry => entry.asset.startsWith('00 Учебник/Assets/Figures/curated/edge-review-2026-10/'));
    expect(entries).toHaveLength(6);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [
        { author: '' }, { modifications: '' }, { license: 'public access' },
        { source_url: 'https://example.org/source' }, { metadata_status: 'unverified' },
        { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { license_evidence_sha256: '0'.repeat(64) }, { derivation: 'redrawn' },
        { crop_box_px: [0, 0, 100, 100] }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
      if (entry.derivation === 'pdf-page-render-crop') {
        expect(() => validateCuratedSourceAsset({ ...entry, source_page: 0 })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...entry, render_page_size_px: [100, 100] })).toThrow();
      }
    }
  });

  it('keeps seven complete MLOps source figures with their distinct licenses', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(entry => entry.asset.startsWith('00 Учебник/Assets/Figures/curated/mlops-review-2026-10/'));
    expect(entries).toHaveLength(7);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [
        { author: '' }, { modifications: '' }, { license: '' },
        { license_url: 'https://example.org/license' },
        { source_url: 'https://example.org/source' },
        { source_image_url: 'https://example.org/image.svg' },
        { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { license_evidence_sha256: '0'.repeat(64) },
        { derivation: 'source-raster-crop' }, { crop_box_px: [0, 0, 100, 100] },
        { metadata_status: 'unverified' }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
      if (entry.asset.endsWith('harvard-dependency.svg')) {
        expect(() => validateCuratedSourceAsset({ ...entry, commit: '0'.repeat(40) })).toThrow();
      }
    }
  });

  it('keeps six complete original Google workflow diagrams and license evidence', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(entry => entry.asset.startsWith('00 Учебник/Assets/Figures/curated/ml-workflow-review-2026-10/'));
    expect(entries).toHaveLength(6);
    for (const entry of entries) {
      validateCuratedSourceAsset(entry);
      for (const invalid of [
        { author: '' }, { modifications: '' }, { license: '' },
        { license_url: 'https://example.org/license' },
        { source_url: 'https://example.org/source' },
        { source_image_url: 'https://example.org/image.png' },
        { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { license_evidence_sha256: '0'.repeat(64) },
        { derivation: 'source-raster-crop' },
        { crop_box_px: [0, 0, 100, 100] },
        { metadata_status: 'unverified' }
      ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
    }
  });

  it('pins all three unmodified CS231n MLP originals to their actual repository license',()=>{
    const rows=JSON.parse(readFileSync(resolve(root,
      '00 Учебник/Assets/Figures/curated/mlp-2026-11-03/assets.json'),'utf8')) as CuratedAsset[];
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets as CuratedAsset[];
    expect(rows).toHaveLength(3);
    for(const row of rows){
      validateCuratedSourceAsset(row);
      expect(registry.find(x=>x.id===row.id)).toEqual(row);
      for(const change of [{license:'open-course-material'},{sha256:'0'.repeat(64)},
        {source_asset_sha256:'0'.repeat(64)},{source_page_sha256:'0'.repeat(64)},
        {license_evidence_sha256:'0'.repeat(64)},{commit:'master'},
        {source_image_url:'https://example.invalid/figure'},
        {provenance_confirmation:'user-confirmed-open-materials'},
        {dimensions_px:[1,1]},{crop_box_px:[0,0,20,20]}])
        expect(()=>validateCuratedSourceAsset({...row,...change})).toThrow();
    }
    expect(readFileSync(resolve(root,'site/public/licenses/cs231n-MIT.txt'),'utf8'))
      .toContain('Copyright (c) 2015 Andrej Karpathy');
  });
  it('keeps complete CS231n loss diagrams pinned without borrowing the license for slide PDFs', () => {
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')) as {assets:CuratedAsset[]};
    const rows=registry.assets.filter(row=>row.asset.includes('/losses-2026-10-23/'));
    expect(rows).toHaveLength(2);
    for(const row of rows){
      validateCuratedSourceAsset(row);
      for(const invalid of [{sha256:'0'.repeat(64)},{source_code_sha256:'0'.repeat(64)},
        {license:'CC BY 4.0'},{author:'Stanford 2025'},{commit:'main'},
        {source_url:'https://example.org/'},{crop_box_px:[0,0,20,20]}])
        expect(()=>validateCuratedSourceAsset({...row,...invalid})).toThrow();
    }
  });

  it('retains five unmodified NumPy tensor figures and rejects provenance drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const rows = registry.assets.filter(row => row.asset.includes('/tensor-foundations-2026-10-23/'));
    expect(rows).toHaveLength(5);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const invalid of [{sha256:'0'.repeat(64)}, {license:'public access'},
        {commit:'main'}, {author:''}, {source_url:'https://example.org/'},
        {crop_box_px:[0,0,20,20]}, {source_code_sha256:'0'.repeat(64)}])
        expect(() => validateCuratedSourceAsset({...row,...invalid})).toThrow();
    }
  });

  it('retains complete versioned Kimi and MaxRL SVGs with distinct source licenses', () => {
    const prefix = '00 Учебник/Assets/Figures/curated/grpo-2026-10-24/';
    const archive = '05 Источники/Originals/grpo-2026-10-24/';
    const specs = JSON.parse(readFileSync(resolve(root, prefix + 'assets.json'), 'utf8')) as CuratedAsset[];
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'));
    const expected = [
      ['kimi-partial-rollouts.svg', '2501.12599v1', 'mshrl-replaybuffer.svg',
        'CC BY-NC-ND 4.0', 'by-nc-nd', 'kimi-v1.html',
        '33af3f50a898c4760743f0caf571d171c0b4de14348cbb54cad262fdcf52a246',
        'c1b9c36460dd2ae8488b78e329384cb3aad4d4a94a4ce41d998193cc7201ae4a',
        '0 0 303.43376 163.86382', 'S2.F3.sf2'],
      ['maxrl-weight-functions.svg', '2602.02710v1', 'weight_functions.svg',
        'CC BY 4.0', 'by', 'maxrl-v1.html',
        '9c3b10853f44bb14651f83b4026a0968fb371e5d6f2e240e4dfeeb1b56dac4cb',
        'e2807b54f56911598301abd26e4549d6c98af3c4f6df632da3c9ec949c03aac0',
        '0 0 716.2461 501.3419', 'S5.F2'],
    ];
    expect(specs).toHaveLength(2);
    for (const [name, paper, sourceName, license, licensePath, html, hash, pageHash, viewBox, figureId] of expected) {
      const spec = specs.find(row => row.asset === prefix + name)!;
      const row = registry.assets.find((entry: CuratedAsset) => entry.id === spec.id);
      expect(row).toEqual(spec);
      validateCuratedSourceAsset(row);
      expect(row.source_url).toBe('https://arxiv.org/abs/' + paper);
      expect(row.source_image_url).toBe('https://arxiv.org/html/' + paper + '/' + sourceName);
      expect(row.source_asset).toBe(row.asset);
      expect(row.sha256).toBe(hash);
      expect(row.source_page_snapshot).toBe(archive + html);
      expect(row.source_page_sha256).toBe(pageHash);
      expect(row.license_evidence).toBe(archive + html);
      expect(row.license_evidence_sha256).toBe(pageHash);
      expect(row.license).toBe(license);
      expect(row.license_url).toBe('https://creativecommons.org/licenses/' + licensePath + '/4.0/');
      const page = readFileSync(resolve(root, archive + html), 'utf8');
      expect(page).toContain('License: ' + license);
      expect(page).toContain('id="' + figureId + '"');
      expect(page).toContain(sourceName);
      const svg = readFileSync(resolve(root, row.asset), 'utf8');
      expect(svg).toContain('viewBox="' + viewBox + '"');
      expect(svg.trimEnd()).toMatch(/<\/svg>$/);
      expect(svg).not.toMatch(/<script\b|<foreignObject\b|\bonload\s*=/i);
      expect(row.crop_box_px).toBeUndefined();
      for (const invalid of [{sha256:'0'.repeat(64)}, {license_evidence_sha256:'0'.repeat(64)},
        {source_url:'https://arxiv.org/abs/' + paper.replace(/v1$/, '')},
        {crop_box_px:[0,0,10,10]}, {source_asset_sha256:'0'.repeat(64)}])
        expect(() => validateCuratedSourceAsset({...row,...invalid})).toThrow();
    }
  });

  it('retains complete Kim-Rush and GKD illustrations with their distinct source evidence', () => {
    const prefix = '00 Учебник/Assets/Figures/curated/distillation-2026-10-25/';
    const archive = '05 Источники/Originals/distillation-2026-10-25/';
    const specs = JSON.parse(readFileSync(resolve(root, prefix+'assets.json'), 'utf8')) as CuratedAsset[];
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'));
    expect(specs).toHaveLength(2);
    for (const spec of specs) {
      const row = registry.assets.find((entry:CuratedAsset) => entry.id === spec.id);
      expect(row).toEqual(spec);
      validateCuratedSourceAsset(row);
      expect(row.license).toBe('CC BY 4.0');
      for (const [file, hash] of [[row.asset,row.sha256], [row.source_asset,row.source_asset_sha256],
        [row.license_evidence,row.license_evidence_sha256]])
        expect(sha256(resolve(root,file))).toBe(hash);
      for (const invalid of [{sha256:'0'.repeat(64)}, {source_asset_sha256:'0'.repeat(64)},
        {license:''}, {source_asset:'../paper.pdf'}])
        expect(()=>validateCuratedSourceAsset({...row,...invalid})).toThrow();
    }
    const svg = specs.find(row=>row.asset.endsWith('.svg'))!;
    expect(svg.source_url).toBe('https://arxiv.org/abs/2306.13649v3');
    expect(svg.sha256).toBe('6c30cdeb8088cb3bfed0685a09097fa0d63297426932b7231efa99ab33ee8b4b');
    expect(svg.crop_box_px).toBeUndefined();
    expect(readFileSync(resolve(root,svg.asset),'utf8')).toContain('viewBox="0 0 687.5 211.2"');
    expect(readFileSync(resolve(root,archive+'gkd-v3.html'),'utf8')).toContain('id="A1.F16"');
    const png = specs.find(row=>row.asset.endsWith('.png'))!;
    expect(png.source_url).toBe('https://aclanthology.org/D16-1139/');
    expect(png.source_page).toBe(3);
    expect(png.source_printed_page).toBe(1319);
    expect(png.render_dpi).toBe(300);
    expect(png.crop_box_px).toEqual([280,280,2270,1360]);
    expect(pngDimensions(resolve(root,png.asset))).toEqual([1990,1080]);
    expect(png.sha256).toBe('ae594f02114520f5af889187b62070ab4f0eb3c71924ed55a3c68db35b9473d0');
  });

  it('replaces missing raw parents for the two legacy synthetic-data originals without changing rights labels', () => {
    const registry = parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'));
    for (const [id,hash,size] of [
      ['curated-nvidia-gcf-000000000035','b5887127ef6ef5a8be95940e78c0ed51284c9d260b0bfdd11b7553971d87b263',[1999,1215]],
      ['curated-olmo2-dolmino-000000000037','653767f5b9910cb2fe39af14c7342273358c158411a485a2b83728ab07b6f66c',[1219,1214]]
    ] as const) {
      const row = registry.assets.find((entry:CuratedAsset)=>entry.id===id);
      expect(row).toBeDefined();
      expect(row.source_asset).not.toMatch(/^raw\//);
      expect(sha256(resolve(root,row.source_asset))).toBe(hash);
      expect(sha256(resolve(root,row.asset))).toBe(hash);
      expect(row.sha256).toBe(hash);
      expect(row.source_asset_sha256).toBe(hash);
      expect(pngDimensions(resolve(root,row.asset))).toEqual(size);
      expect(row.source_image_url).toMatch(/^https:\/\//);
      expect(row.license).toBe('open educational material; reuse confirmed by repository owner');
      expect(row.provenance_confirmation).toBe('user-confirmed-open-materials');
      validateCuratedSourceAsset(row);
      expect(()=>validateCuratedSourceAsset({...row,source_asset_sha256:'0'.repeat(64)})).toThrow();
      expect(()=>validateCuratedSourceAsset({...row,source_asset:'raw/papers/missing.png'})).toThrow();
    }
  });

  it('uses the higher-resolution prefill original and tracked parents for the three paging figures', () => {
    const registry = parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'));
    const chapter = '00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md';
    const prefill = registry.assets.find((row:CuratedAsset)=>row.id==='curated-attention-947a70b7cef4');
    expect(prefill.used_in).toContain(chapter);
    expect(prefill.license).toBe('CC BY 4.0');
    expect(prefill.author).toContain('Baolin Li');
    expect(prefill.source_url).toBe('https://arxiv.org/abs/2407.12391v1');
    expect(sha256(resolve(root,prefill.asset))).toBe('947a70b7cef4d1875d295692731cd4216f189daaf5ac7b586106fc542626e3a6');
    expect(pngDimensions(resolve(root,prefill.asset))).toEqual([899,344]);
    expectDecodableImage(resolve(root,prefill.asset));
    const legacy = registry.assets.find((row:CuratedAsset)=>row.id==='curated-cs336-prefill-00000000003e');
    expect(legacy.used_in).not.toContain(chapter);
    expect(existsSync(resolve(root,legacy.asset))).toBe(true);
    for (const [name,hash] of [
      ['fragmentation','43fdea6749242beb1b07c82bc7d4e617012e485271f284093c94c198b410fe41'],
      ['blocks','16b5c731f1f5ee4e031b5cbbce0ca97a4df580b63864de1dd08fde82dc605f6d'],
      ['logical','8caaf998a04761655545fa4fc33b99cb4874d258f426c858609f5e4a54b38c9c']
    ]) {
      const row = registry.assets.find((entry:CuratedAsset)=>entry.asset===
        '00 Учебник/Assets/Figures/curated/topics-53-59-source-first/paged-attention-'+name+'.png');
      validateCuratedSourceAsset(row);
      expect(row.source_asset).toBe('05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/images/paged-attention-'+name+'.png');
      expect(row.source_image_url).toContain(cs336Commit);
      expect(row.sha256).toBe(hash);
      expect(row.source_asset_sha256).toBe(hash);
      expect(row.license).toBe('CC BY 4.0');
      expect(row.crop_box_px).toBeUndefined();
      expect(readFileSync(resolve(root,row.asset))).toEqual(readFileSync(resolve(root,row.source_asset)));
    }
  });

  it('pins all six unchanged HF decoding originals without missing raw parents or new license assertions', () => {
    const registry = parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'));
    const prefix = '00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-';
    for (const [name,hash] of [
      ['greedy_search','1f4390b7ada9e92902d379ed61c80a0966d95f1a5c3b7156792544780aed7b19'],
      ['beam_search','e1db549ab759682f802dbc4dbf6e3d357bafef10f71f300069cdd583eafe568a'],
      ['sampling_search','ed8b6920d219c2a32d47a7ce8bfc28d434cf306ad44792d59b37ba4015361edd'],
      ['sampling_search_with_temp','816eab79cdfd4efa9966ef3e4fc1a9071a0ea227bf2b483935462d320aceb250'],
      ['top_k_sampling','bf7c378f0879ff8acb31af88adbab0cfff411addd1e1f0e2c8f544c5182c847d'],
      ['top_p_sampling','6cdf2ce91023691463116196e2a27629344c497a14660df76d5cf3e49eda6931']
    ]) {
      const row = registry.assets.find((entry:CuratedAsset)=>entry.asset===prefix+name+'.png');
      expect(row).toBeDefined();
      validateCuratedSourceAsset(row);
      expect(row.source_asset).toBe(row.asset);
      expect(row.sha256).toBe(hash);
      expect(row.source_asset_sha256).toBe(hash);
      expect(sha256(resolve(root,row.asset))).toBe(hash);
      expect(row.source_image_url).toBe('https://huggingface.co/blog/assets/02_how-to-generate/'+name+'.png');
      expect(row.license).toBe('open educational material; reuse confirmed by repository owner');
      expect(row.provenance_confirmation).toBe('user-confirmed-open-materials');
    }
  });

  it('pins complete USC and sequential-parallel originals without redraw or crop', () => {
    const prefix = '00 Учебник/Assets/Figures/curated/test-time-compute-2026-10-25/';
    const archive = '05 Источники/Originals/test-time-compute-2026-10-25/';
    const specs = JSON.parse(readFileSync(resolve(root,prefix+'assets.json'),'utf8')) as CuratedAsset[];
    const registry = parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'));
    expect(specs).toHaveLength(2);
    const expected = [
      ['usc-workflow.png','2311.17311v1','figs/usc.png','usc-v1.html',[933,181],
        'fed9bdb8739a8817279c8d1678516ba348339af451f843019033b81ef329dc17'],
      ['snell-sequential-parallel-ratio.png','2408.03314v1',
        'revisions_varing_ratio_and_with_difficulty.png','snell-v1.html',[2034,710],
        'eaac91cacfb34c2d846e1f2a85f697b7d716f11f4576bfad4c4b37ae606ec3ae']
    ] as const;
    for (const [name,paper,imageName,html,size,hash] of expected) {
      const spec = specs.find(row=>row.asset===prefix+name)!;
      const row = registry.assets.find((entry:CuratedAsset)=>entry.id===spec.id);
      expect(row).toEqual(spec);
      validateCuratedSourceAsset(row);
      expect(row.source_url).toBe('https://arxiv.org/abs/'+paper);
      expect(row.source_image_url).toBe('https://arxiv.org/html/'+paper+'/'+imageName);
      expect(row.source_asset).toBe(row.asset);
      expect(row.sha256).toBe(hash);
      expect(pngDimensions(resolve(root,row.asset))).toEqual(size);
      expect(row.crop_box_px).toBeUndefined();
      expect(row.license).toBe('CC BY 4.0');
      const source = readFileSync(resolve(root,archive+html),'utf8');
      expect(source).toContain('License: CC BY 4.0');
      expect(source).toContain(imageName);
      for (const [path,digest] of [[row.asset,row.sha256],
        [row.license_evidence,row.license_evidence_sha256]])
        expect(sha256(resolve(root,path))).toBe(digest);
      for (const invalid of [{sha256:'0'.repeat(64)}, {license:''},
        {source_asset_sha256:'0'.repeat(64)}, {crop_box_px:[0,0,10,10]}])
        expect(()=>validateCuratedSourceAsset({...row,...invalid})).toThrow();
    }
  });

  it('pins the complete reproduced R1 pipeline to its actual PDF and crop', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'));
    const row = registry.assets.find((entry: CuratedAsset) => entry.id === 'curated-deepseek-r1-51d3877bb09b');
    validateCuratedSourceAsset(row);
    expect(row.source_asset).toBe('05 Источники/Papers/Source PDFs/deepseek-r1-v2.pdf');
    expect(row.source_page).toBe(6);
    expect(row.render_dpi).toBe(300);
    expect(row.render_page_size_px).toEqual([2481, 3508]);
    expect(row.crop_box_px).toEqual([250, 280, 2150, 1220]);
    expect(pngDimensions(resolve(root, row.asset))).toEqual([1900, 940]);
    expect(row.sha256).toBe('51d3877bb09b8723dfa71d67098ae5cf6782caf32caa23536f0eb95876ac0614');
    expect(row.license).toBe('CC BY 4.0');
    expect(row.license_url).toBe('https://www.nature.com/articles/s41586-025-09422-z#rightslink');
  });

  it('protects the visually reviewed MML crops including axes, input and parameter labels', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const archivedFigures = ['retired-derivative-figures.json', 'retired-probability-figures.json']
      .flatMap(name => JSON.parse(readFileSync(resolve(root,
        '05 Источники/Originals/mml-source-2026-10-30/' + name), 'utf8')).figures);
    const prefix = '00 Учебник/Assets/Figures/curated/foundations-source-first/';
    const cases = [
      ['mml-matrix-gradient-shape.png', 161, 155, [560, 486, 1460, 1560],
        '5d0136507298b3d21643c50fc1360a0e760dc5c5a7d5ccf12a29fc7aa8d0b9a0'],
      ['mml-matrix-gradient-flatten.png', 161, 155, [550, 1615, 1470, 2615],
        '27c461fc25326b1934d85ea2e7900004f0839894fb9249310ba3d1dbf879fc7b'],
      ['mml-computation-graph-forward.png', 166, 160, [580, 500, 2020, 865],
        '79fb5bb7d1ac2a60b76a49ccc16ac65371d80d7f94dbdfacf64f7decd0c3e9d5'],
      ['mml-discrete-continuous-density.png', 188, 182, [600, 495, 2010, 1000],
        '49e803f4c725f14a81ee4475450f2ea94fe5d6427779db14e914918907d48dd0']
    ] as const;
    for (const [name, physical, printed, box, hash] of cases) {
      const archived = archivedFigures.find((item: { from: string }) => item.from === prefix + name);
      const row = archived?.original_registry_record;
      expect(row).toBeDefined();
      expect(registry.assets.some(entry => entry.asset === prefix + name)).toBe(false);
      expect(existsSync(resolve(root, prefix + name))).toBe(false);
      expect(row.source_page).toBe(physical);
      expect(row.source_printed_page).toBe(printed);
      expect(row.render_dpi).toBe(300);
      expect(row.render_page_size_px).toEqual([2481, 3508]);
      expect(row.crop_box_px).toEqual(box);
      expect(row.source_asset_sha256).toBe('3f87b70c64a35d30ec4e565b4d5cfd18fe913002d3b3bed44691c7d0838911bc');
      expect(row.sha256).toBe(hash);
      const retainedPath = archived.to;
      expect(sha256(resolve(root, retainedPath))).toBe(hash);
      expect(pngDimensions(resolve(root, retainedPath))).toEqual([box[2]-box[0], box[3]-box[1]]);
    }
  });

  it('retains the original D2L frequency and Gaussian plots with their license', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const row = registry.assets.find(entry => entry.asset.endsWith('/d2l-coin-frequency-convergence.svg'))!;
    validateCuratedSourceAsset(row);
    expect(row.sha256).toBe('2906234b60991e9cf930f345eb5948caae3e810992622a61e2e8cd7db7654dfd');
    expect(() => validateCuratedSourceAsset({ ...row, license: 'MIT' })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...row, sha256: '0'.repeat(64) })).toThrow();
    const gaussian = registry.assets.find(entry => entry.asset.endsWith('/d2l-gaussian-densities.svg'))!;
    validateCuratedSourceAsset(gaussian);
    expect(gaussian.sha256).toBe('6d5c7c46c781ed670a49a8715fe036f19651f33317aaca733d128a7f0dd68cd8');
  });

  it('pins the recovered CS231n source PDF and exact full-slide pixel provenance without inventing a license', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'));
    for (const id of ['curated-cs231n-softmax-ce-0000000000a6',
      'curated-cs231n-boundaries-0000000000a7']) {
      const entry = registry.assets.find((row: CuratedAsset) => row.id === id);
      expect(entry).toBeDefined();
      expect(entry.metadata_status).toBe('verified');
      expect(entry.derivation).toBe('pdf-full-slide-render-verified');
      expect(entry.source_asset).toContain('cs231n-provenance-2026-10-29/lecture_2.pdf');
      expect(entry.rights_status).toBe('course-slide-republication-unverified');
      expect(sha256(resolve(root, entry.asset))).toBe(entry.sha256);
      validateCuratedSourceAsset(entry);
      for (const invalid of [
        {metadata_status: 'needs-confirmation'}, {source_asset: 'missing-source.pdf'},
        {sha256: '0'.repeat(64)}, {rights_status: 'source-license-verified'},
        {license_url: 'https://cs231n.stanford.edu/'}, {used_in: ['00 Учебник/00 Математические и ML-основания/04 Функции потерь.md']},
        {author: 'Fei-Fei Li; Ranjay Krishna; Danfei Xu'},
        {source_url: 'https://example.org/slide.pdf'}
      ]) expect(() => validateCuratedSourceAsset({...entry,...invalid})).toThrow();
    }
  });

  it('preserves the three new original split diagrams and rejects provenance drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const entries = registry.assets.filter(row => /^curated-split-/.test(row.id ?? ''));
    expect(entries).toHaveLength(3);
    for (const row of entries) {
      validateCuratedSourceAsset(row);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_url: 'https://example.org' }, { license: 'public access' },
        { crop_box_px: [0, 0, 100, 100] }, { license_evidence_sha256: '0'.repeat(64) }])
        expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });

  it('preserves the original MOOC polynomial fits and resampled-model SVGs', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const prefix = '00 Учебник/Assets/Figures/curated/foundations-source-first/';
    const cases = [
      ['mooc-polynomial-fits.svg', 'polynomial_overfit.svg', '5dd540581db90033222781e858036d226d3683122a0ae9b00572e7764e879240'],
      ['mooc-underfit-resamples.svg', 'polynomial_underfit_resample_all.svg', 'a3c2ea999f924de75dce9250c26b632442c95eb34d050c891d100f2ebbc4a1eb'],
      ['mooc-overfit-resamples.svg', 'polynomial_overfit_resample_all.svg', '6e8a65e19e4bbd8ffcde37783b2aeda480c990f22d874e3717db4dd6d1aaa407'],
    ] as const;
    for (const [name, original, hash] of cases) {
      const row = registry.assets.find(entry => entry.asset === prefix + name)!;
      validateCuratedSourceAsset(row);
      expect(row.commit).toBe('0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6');
      expect(row.source_url).toBe('https://github.com/INRIA/scikit-learn-mooc/blob/' + row.commit + '/figures/' + original);
      expect(row.source_image_url).toBe('https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/' + row.commit + '/figures/' + original);
      expect(row.sha256).toBe(hash);
      expect(sha256(resolve(root, row.asset))).toBe(hash);
      expect(readFileSync(resolve(root, row.asset)).equals(readFileSync(resolve(root, 'Assets/Sources/Scikit-learn MOOC/figures/' + original)))).toBe(true);
      expect(row.license).toBe('CC BY 4.0');
      expect(row.license_url).toBe('https://github.com/INRIA/scikit-learn-mooc/blob/' + row.commit + '/LICENSE');
      expect(sha256(resolve(root, row.license_evidence!))).toBe(row.license_evidence_sha256);
      expect(row.crop_box_px).toBeUndefined();
    }
  });

  it('preserves the four original Lloyd-step SVGs and their author license grants', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const figures = registry.assets.filter(row => row.provenance_confirmation === 'commons-lloyd-source-snapshot');
    expect(figures).toHaveLength(4);
    for (const row of figures) {
      validateCuratedSourceAsset(row);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_page_sha256: '0'.repeat(64) }, { license_evidence_sha256: '0'.repeat(64) },
        { author: 'Google' }, { license: 'CC BY 4.0' }, { derivation: 'source-svg-crop' },
        { crop_box_px: [0, 0, 10, 10] }, { source_image_url: 'https://example.org/image.svg' }])
        expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });

  it('validates original clustering plots, reachability diagrams and licenses', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    for (const name of ["elbow.png","silhouette.png","shapes-kmeans.png","shapes-hdbscan.png","dbscan-core-border.png","hdbscan-mst.png","hdbscan-condensed-selected.png","hdbscan-mutual-reachability.svg"]) {
      const row = registry.assets.find(entry => entry.asset === '00 Учебник/Assets/Figures/curated/classical-ml-clustering/' + name)!;
      validateCuratedSourceAsset(row);
      for (const invalid of [{sha256:'0'.repeat(64)}, {source_page_sha256:'0'.repeat(64)},
        {license:'MIT'}, {crop_box_px:[0,0,10,10]}, {source_image_url:'https://example.org/fake.png'}])
        expect(() => validateCuratedSourceAsset({...row,...invalid})).toThrow();
    }
  });

  it('preserves original grouped and temporal CV gallery figures with pinned code', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    for (const name of ['group-kfold.png', 'time-series-split.png']) {
      const row = registry.assets.find(entry => entry.asset === '00 Учебник/Assets/Figures/curated/classical-ml-evaluation/' + name)!;
      validateCuratedSourceAsset(row);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_page_sha256: '0'.repeat(64) }, { source_code_sha256: '0'.repeat(64) },
        { source_code_url: 'https://example.org/source.py' }, { commit: '0'.repeat(40) },
        { license: 'MIT' }, { crop_box_px: [0, 0, 10, 10] },
        { source_image_url: 'https://example.org/source.png' }])
        expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });

  it('keeps evaluation panels tied to source pixels, licenses and snapshots', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    for (const name of ["pr-curve.png","roc-curve.png","pr-roc.png","calibration-curve.png","calibration-legend.png","calibration.png","regression-actual.png","regression-residual.png","regression-residuals.png","nested-first-outer.png","nested-legend.png"]) {
      const row = registry.assets.find(entry => entry.asset === '00 Учебник/Assets/Figures/curated/classical-ml-evaluation/' + name)!;
      validateCuratedSourceAsset(row);
      for (const invalid of [{sha256: '0'.repeat(64)}, {source_asset_sha256: '0'.repeat(64)},
        {license: 'MIT'}, {source_image_url: 'https://example.org/image.png'},
        {crop_box_px: [0,0,10,10]}, {license_evidence_sha256: '0'.repeat(64)}])
        expect(() => validateCuratedSourceAsset({...row, ...invalid})).toThrow();
    }
  });

  it('pins the original news-metric and RFM course figures and rejects metadata drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), {merge:true}) as {assets:CuratedAsset[]};
    for (const name of ['mooc-news-external-metrics.png', 'mooc-news-silhouette.png', 'mooc-rfm-quantile-layers.png']) {
      const row = registry.assets.find(entry => entry.asset.endsWith('/' + name))!;
      expect(() => validateCuratedSourceAsset(row)).not.toThrow();
      for (const mutation of [{sha256:'0'.repeat(64)}, {commit:'main'}, {license:'unverified'},
        {source_image_url:'https://example.org/image.png'}, {crop_box_px:[0,0,10,10]},
        {license_evidence_sha256:'0'.repeat(64)}, {metadata_status:'needs-confirmation'}])
        expect(() => validateCuratedSourceAsset({...row,...mutation})).toThrow();
    }
  });

  it('pins the post-transform regression original and rejects metadata drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), {merge:true}) as {assets:CuratedAsset[]};
    const row = registry.assets.find(entry => entry.asset.endsWith('/regression-transformed-target.png'))!;
    expect(() => validateCuratedSourceAsset(row)).not.toThrow();
    for (const mutation of [{sha256:'0'.repeat(64)}, {commit:'main'}, {license:'unverified'},
      {source_image_url:'https://example.org/image.png'}, {crop_box_px:[0,0,10,10]},
      {license_evidence_sha256:'0'.repeat(64)}, {metadata_status:'needs-confirmation'}])
      expect(() => validateCuratedSourceAsset({...row,...mutation})).toThrow();
  });

  it('keeps ensemble illustrations tied to their actual algorithms and source pixels', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const prefix = '00 Учебник/Assets/Figures/curated/classical-ml-ensembles/';
    const cases = [["bagging-hundred-average.png","71b3ddc7bcc100af144a93c00a8e87a7f665b9dcd4532b73c3c42793107135c1"],["bagging-three-average.png","935b6fa4b6fb39d7827b96ecffe783e9b03653c486794aa1470830b955897b8d"],["bagging-three-trees.png","aa2883cee12aed2a188a999cf379dc876ac90dad773b9870ca12f448cc464129"],["bootstrap-sample.png","de65309953f108ee9132d1c3326a630102d4d91237c24de9010b6a5a375ada3f"],["ensemble-boosting1.svg","b0313183ac6692c9fdba2cd245f633fc6063b84b17becc02fd57028b6fbf55e4"],["ensemble-boosting2.svg","d303ae78da67b4cf5dd3095bafba754e8e3666073bab3314a3828312f6010c81"],["ensemble-boosting3.svg","adbc3245c5d493e82ab1e8c4fa16d73d456da993954f1dc54de52dbbb0e0493a"],["gradient-fit-residuals.png","a4a3778869152286bcd5c54c53631d208f0896259b4486841ded4f0f91dfaff2"],["gradient-initial-legend.png","ef42284c96b1ce1f547a9d1a28879f6de4f56dfc53625f9e9c724dd9d02cbef7"],["gradient-initial-plot.png","f52097fa8531817dc25e009c14a94ec19ecd0c5d9ba26024d94279d336b615d0"],["gradient-initial-residuals.png","9a116965bb8133ee4bfe4d8d67625b2e52709f10e94cf871655ba041d8f50cd5"],["gradient-residual-legend.png","f3a8e811ef6cb09f5e08a19bd456538b99105da8987d36facb67ac3641fe1de5"],["gradient-residual-plot.png","5a3fe3b249365d8aa5283676cd151a5268eb3df6feddacb99f0e220166fc757b"]];
    for (const [name, hash] of cases) {
      const row = registry.assets.find(entry => entry.asset === prefix + name)!;
      validateCuratedSourceAsset(row);
      expect(row.sha256).toBe(hash);
      expect(sha256(resolve(root, row.asset))).toBe(hash);
      if (name.endsWith('.svg')) {
        expect(row.commit).toBe('0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6');
        expect(readFileSync(resolve(root, row.asset)).equals(readFileSync(resolve(root, 'Assets/Sources/Scikit-learn MOOC/figures/' + name.replace('ensemble-', ''))))).toBe(true);
      } else {
        for (const invalid of [
          { sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
          { source_page_sha256: '0'.repeat(64) }, { license: 'MIT' },
          { crop_box_px: [0,0,10,10] }, { source_image_url: 'https://example.org/img.png' }
        ]) expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
      }
    }
  });

  it('validates the original MOOC penguin tree and leaf-frequency plots', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    for (const name of ['penguins-tree.png', 'penguins-probabilities.png']) {
      const row = registry.assets.find(entry => entry.asset === '00 Учебник/Assets/Figures/curated/classical-ml-trees/' + name)!;
      validateCuratedSourceAsset(row);
      for (const invalid of [
        { sha256: '0'.repeat(64) }, { source_page_sha256: '0'.repeat(64) },
        { source_asset_sha256: '0'.repeat(64) }, { license: 'MIT' },
        { source_image_url: 'https://example.org/image.png' },
        { derivation: 'source-raster-crop' }
      ]) expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });

  it('preserves the decision-tree course illustrations with pinned attribution', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const prefix = '00 Учебник/Assets/Figures/curated/classical-ml-trees/';
    const cases = [["dt_fit.svg","affb65f41fc0e8954679cf6692a77b04e02aefd0ad835507b8fd42f2b49c8e3d"],["dt_overfit.svg","2bdbadd5ad5271fd36469e14bde2d53bff64f6de2802e0debec7af54e425b280"],["dt_underfit.svg","1196fe3b15c472a52dfd916fb94797a2395ce067adcb09f5ead986409b04b9c7"],["tree2D_1split.svg","5d9f496904f83c14f9753a5a90341cc6c3b5f3977cce55dbd44d76462c8bbf66"],["tree2D_2split.svg","2b0c4f5a4096f7e228e90a599633fbb0c95d62f26ed2aa717266fe6c51440481"],["tree2D_3split.svg","a785875b014599e49ecefe2dee03d03896c4e47ded3eb3696b9fadf187837cdd"],["tree_example.svg","57cad9e1f19f0134cbfa9adaa0472d97f363dd50fc1bac8a07e3e14c756bbc8e"],["tree_regression1.svg","cc0e7197c853e6583debecd51f71d804406be34d56eb668964016a40602c4218"],["tree_regression2.svg","cd10a38cbb4f09f08e5ade91554d23ee944e5b487c370a8af6dad74e2b3fc7c9"],["tree_regression4.svg","92ace7627ff8ae2e520d1d512fe3e60d5038f1a74c0dcf0f851439b140ab6322"]];
    for (const [name, hash] of cases) {
      const row = registry.assets.find(entry => entry.asset === prefix + name)!;
      validateCuratedSourceAsset(row);
      expect(row.commit).toBe('0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6');
      expect(row.source_url).toBe('https://github.com/INRIA/scikit-learn-mooc/blob/' + row.commit + '/figures/' + name);
      expect(row.source_image_url).toBe('https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/' + row.commit + '/figures/' + name);
      expect(row.sha256).toBe(hash);
      expect(sha256(resolve(root, row.asset))).toBe(hash);
      expect(readFileSync(resolve(root, row.asset)).equals(readFileSync(resolve(root, 'Assets/Sources/Scikit-learn MOOC/figures/' + name)))).toBe(true);
      expect(row.license).toBe('CC BY 4.0');
      expect(sha256(resolve(root, row.license_evidence!))).toBe(row.license_evidence_sha256);
      expect(row.crop_box_px).toBeUndefined();
    }
    const tight = registry.assets.find(entry => entry.asset === prefix + 'tree-example-tight.svg')!;
    validateCuratedSourceAsset(tight);
    expect(tight.source_asset).toBe(prefix + 'tree_example.svg');
    expect(tight.sha256).toBe('661c2498951dc9d39761e433cd1f8761b369d6154aaaefc04bea223990065d43');
    expect(readFileSync(resolve(root, tight.asset), 'utf8')).toContain('viewBox="230 92 540 402"');
    const permutation = registry.assets.find(entry => entry.asset === prefix + 'permutation-predictive-feature.png')!;
    validateCuratedSourceAsset(permutation);
    expect(permutation.commit).toBe('866c0f51e7560ef0303cbcc5f159df5382ea9e3f');
    expect(permutation.sha256).toBe('e3fccea5418c250b6105bbc54549ee40183ae0e928d9074123d057e925bb9731');
    expect(pngDimensions(resolve(root, permutation.asset))).toEqual([845,439]);
    expect(permutation.license).toBe('BSD-3-Clause');
    expect(permutation.license_url).toBe('https://github.com/scikit-learn/scikit-learn/blob/' + permutation.commit + '/COPYING');
    expect(readFileSync(resolve(root, permutation.source_page_snapshot!), 'utf8')).toContain('../images/permuted_predictive_feature.png');
    for (const invalid of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
      { source_page_sha256: '0'.repeat(64) }, { license_evidence_sha256: '0'.repeat(64) },
      { source_image_url: 'https://example.org/unrelated.png' }, { crop_box_px: [0,0,100,100] }])
      expect(() => validateCuratedSourceAsset({ ...permutation, ...invalid })).toThrow();
  });

  it('keeps linear-model course plots and the actual spline basis with source evidence', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const prefix = '00 Учебник/Assets/Figures/curated/classical-ml-linear/';
    const cases = [
      ["mooc-residuals.svg","linear_fit_red.svg","efd859918cb76b97df6af164e13068f01052a4f591816b2841462be843fd4397"],
      ["mooc-regression-plane.svg","lin_reg_3D.svg","8a8c3c81e8c40ddd2c00ea0db4db7d1933d06171acc0b9f40e6754cac11f838f"],
      ["mooc-sigmoid.svg","logistic_color.svg","de448ccca81b5e44ca04824f01ddec954d1c139c428438435e967d0755985649"],
      ["mooc-logistic-surface.svg","logistic_3D.svg","a6e36b909171b66829b9ae20849b031e048a17a6a82e201fe2f8569500834cf6"],
      ["mooc-multiclass.svg","multinomial.svg","08ba1e0af9c6b68a63c935eccd759a94871681f6a468b5f07e913a35b1d0cb8c"],
      ["mooc-nonseparable.svg","lin_not_separable.svg","380adc120ef7e8e0abedc75bc80e224fd9778fabff7a6133bef040d6c6e88f6e"],
      ["mooc-ridge-0.svg","ridge_alpha_0.svg","b6fd7aee6c35e8b5f3fe61f4bf006e5bbdf15722d02d6fb83776c7aaaf094b7e"],
      ["mooc-ridge-50.svg","ridge_alpha_50.svg","146844c546b1177dbbb1629329147f2e18b8a7edfac532bbfd29a696b1913c65"],
      ["mooc-ridge-500.svg","ridge_alpha_500.svg","6bf7070563ace5c3c215890008cc8161d0163ee8699101d8abcddea9ab81b1af"],
      ["mooc-interpolation-overfit.svg","linear_splines.svg","df27e8dbf0d140c47b25ccc1b99814394d7d697a27b3c4ea317252b2e14c22ed"]
    ];
    for (const [local, original, hash] of cases) {
      const row = registry.assets.find(entry => entry.asset === prefix + local)!;
      validateCuratedSourceAsset(row);
      expect(row.commit).toBe('0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6');
      expect(row.source_image_url).toBe('https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/' + row.commit + '/figures/' + original);
      expect(sha256(resolve(root, row.asset))).toBe(hash);
      expect(row.sha256).toBe(hash);
      // Native equality checks the exact bytes without walking a large SVG's
      // Buffer as millions of JavaScript object properties.
      expect(readFileSync(resolve(root, row.asset)).equals(
        readFileSync(resolve(root, 'Assets/Sources/Scikit-learn MOOC/figures/' + original)))).toBe(true);
      expect(row.license).toBe('CC BY 4.0');
      expect(sha256(resolve(root, row.license_evidence!))).toBe(row.license_evidence_sha256);
    }
    const basis = registry.assets.find(entry => entry.asset === prefix + 'sklearn-polynomial-spline-basis.png')!;
    validateCuratedSourceAsset(basis);
    for (const name of ['sklearn-polynomial-basis.png', 'sklearn-spline-basis.png']) {
      const crop = registry.assets.find(entry => entry.asset === prefix + name)!;
      validateCuratedSourceAsset(crop);
      expect(() => validateCuratedSourceAsset({ ...crop, crop_box_px: [0,0,100,100] })).toThrow();
    }
    for (const invalid of [{ sha256: '0'.repeat(64) }, { license: 'public access' },
      { source_page_sha256: '0'.repeat(64) }, { source_code_sha256: '0'.repeat(64) },
      { source_image_url: 'https://example.org/unrelated.png' }, { crop_box_px: [0,0,100,100] }])
      expect(() => validateCuratedSourceAsset({ ...basis, ...invalid })).toThrow();
  });

  it('preserves the reviewed MOOC pipeline figures and their pinned license', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const prefix = '00 Учебник/Assets/Figures/curated/classical-ml-pipeline/';
    const cases = [
      ["mooc-predictor-fit.svg","api_diagram-predictor.fit.svg","1ca742d7446bbac56f55a109f58cc4e311aa57f5f7ac06b43a82d6a4645f52ec"],
      ["mooc-predictor-predict.svg","api_diagram-predictor.predict.svg","fd06ffe2fca74105bf79e0d1a9e30d4af1b2c2a80d58e3e25556f3c2f3045040"],
      ["mooc-predictor-score.svg","api_diagram-predictor.score.svg","df6d93fb0a2fec090df97af8805507a2fe2a8635b960c1882496d6f6e8c227cc"],
      ["mooc-transform.svg","api_diagram-transformer.transform.svg","2050af0970beee2f4c9dacbf672919a15325b2673d29cf2a997c96ba1f5b4ffe"],
      ["mooc-pipeline-fit.svg","api_diagram-pipeline.fit.svg","3a4c257d2d9ccd00942d707b4e4d99830142ddff8ffbccb3b71bd4b020adc5e2"],
      ["mooc-pipeline-predict.svg","api_diagram-pipeline.predict.svg","a19224b3d3f4bc01bdd5f5a550f2a37ba871f962e888b7e7e3122c780eb50b77"],
      ["mooc-columns.svg","api_diagram-columntransformer.svg","77115899eb90b613d7c4bba8efc9b1bb71a65c4435fa15a0bf74d0d3ac4d12d7"]
    ];
    for (const [local, original, hash] of cases) {
      const row = registry.assets.find(entry => entry.asset === prefix + local)!;
      validateCuratedSourceAsset(row);
      expect(row.commit).toBe('0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6');
      expect(row.source_url).toBe('https://github.com/INRIA/scikit-learn-mooc/blob/' + row.commit + '/figures/' + original);
      expect(row.source_image_url).toBe('https://raw.githubusercontent.com/INRIA/scikit-learn-mooc/' + row.commit + '/figures/' + original);
      expect(row.sha256).toBe(hash);
      expect(sha256(resolve(root, row.asset))).toBe(hash);
      expect(readFileSync(resolve(root, row.asset)).equals(readFileSync(resolve(root, 'Assets/Sources/Scikit-learn MOOC/figures/' + original)))).toBe(true);
      expect(row.license).toBe('CC BY 4.0');
      expect(sha256(resolve(root, row.license_evidence!))).toBe(row.license_evidence_sha256);
      expect(row.crop_box_px).toBeUndefined();
    }
  });

  it('keeps the sharper CS231n note diagram unchanged and retains its MIT notice', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const row = registry.assets.find(entry => entry.asset.endsWith('/cs231n-pixelspace-original.jpeg'))!;
    validateCuratedSourceAsset(row);
    expect(row.commit).toBe('8ec57ccfc32379ae4b37c45de949af359673f92f');
    expect(row.license).toBe('MIT');
    expect(sha256(resolve(root, row.asset))).toBe('311ed25c3a87f5299d5b08ed33763b6764d3039ab5bac3f2d7b1422c50bad289');
    expect(sha256(resolve(root, row.license_evidence!))).toBe(row.license_evidence_sha256);
    expect(readFileSync(resolve(root, row.license_evidence!), 'utf8')).toContain('Copyright (c) 2015 Andrej Karpathy');
  });

  it('pins CLIP and SigLIP paper Figure 1 and keeps the former course copies out of the site', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const current = registry.assets.filter((row: CuratedAsset) =>
      row.asset.startsWith('00 Учебник/Assets/Figures/curated/vision-objectives-2026-11-09/'));
    expect(current.map((row: CuratedAsset) => row.id).sort()).toEqual([
      'curated-clip-paper-figure1-351759e7ee8b',
      'curated-siglip-paper-figure1-6e373e662574',
    ]);
    for (const row of current) validateCuratedSourceAsset(row, root);
    for (const id of ['curated-stanford-cs336-multimodal-siglip-parallelism-21fbe73a2d33',
      'curated-vlm2026-clip-figure1-ac050e1cf247']) {
      const previous = registry.assets.find((row: CuratedAsset) => row.id === id);
      expect(previous).toMatchObject({ publication_status: 'retired', used_in: [] });
      expect(sha256(resolve(root, previous.asset))).toBe(previous.sha256);
    }
  });

  it('pins the original LLaVA paper Figure 1 and retires the course copy', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const row = registry.assets.find(entry => entry.id === 'curated-llava-paper-figure1-310746447f5a')!;
    validateCuratedSourceAsset(row);
    const previous = registry.assets.find(entry => entry.id === 'curated-vlm2026-llava-figure1-27391332e106')!;
    expect(previous).toMatchObject({ publication_status: 'retired', used_in: [] });
    expect(sha256(resolve(root, previous.asset))).toBe(previous.sha256);
  });

  it('pins the complete unmodified OneVision AnyRes original and retains the course copy in the archive', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    validateCuratedSourceAsset(registry.assets.find(entry => entry.id === 'curated-onevision-higher-anyres-db33ded7e7e6')!);
    const previous = registry.assets.find(entry => entry.id === 'curated-stanford-cs336-multimodal-llava-anyres-ba1d449afcd3')!;
    expect(previous).toMatchObject({ publication_status: 'retired', used_in: [] });
    expect(sha256(resolve(root, previous.asset))).toBe(previous.sha256);
  });

  it('pins both original Prismatic training-ablation figures from the PMLR PDF', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    for (const id of ['curated-prismatic-stage-ablation-49aca74ebdc0', 'curated-prismatic-vision-ablation-aa5f932da538'])
      validateCuratedSourceAsset(registry.assets.find(entry => entry.id === id)!);
  });

  it('preserves the reviewed source figures and keeps the three superseded copies archival', () => {
    const rows = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets as CuratedAsset[];
    for (const spec of finalSourceFigures) {
      const row = rows.find(candidate => candidate.asset === spec.asset)!;
      validateCuratedSourceAsset(row);
      expect(() => validateCuratedSourceAsset({ ...row, license: 'Unverified' })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...row, crop_box_px: [0, 0, 2, 2] })).toThrow();
    }
    for (const spec of finalRetiredFigures) {
      const row = rows.find(candidate => candidate.asset === spec.asset)!;
      expect(row).toMatchObject({ ...spec, used_in: [], publication_status: 'retired', metadata_status: 'needs-confirmation' });
      expect(sha256(resolve(root, spec.asset))).toBe(spec.sha256);
      for (const note of spec.former_used_in) {
        const body = readFileSync(resolve(root, note), 'utf8');
        expect(body).not.toContain(spec.asset);
        expect(body).toContain(spec.replacement);
      }
      expect(spec.former_used_in.some(note => readFileSync(resolve(root, note), 'utf8').includes(spec.figure_url))).toBe(true);
    }
  });
  it('links all six retired ASTRA and ToolRL figures to the correct source panels', () => {
    const assets = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets as CuratedAsset[];
    for (const spec of archivedAgentPaperFigures) {
      const row = assets.find(candidate => candidate.asset === spec.asset)!;
      validateCuratedSourceAsset(row);
      const body = readFileSync(resolve(root, `Papers/${spec.owner}.md`), 'utf8');
      expect(body).toContain(spec.figure_url);
      expect(body).not.toContain(spec.asset);
      expect(() => validateCuratedSourceAsset({ ...row, used_in: [`Papers/${spec.owner}.md`] })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...row, license: 'Apache-2.0' })).toThrow();
    }
  });
  it('keeps the three retired paper copies unverified and byte-pinned', () => {
    const rows=(parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')) as {assets:CuratedAsset[]}).assets;
    expect(retiredFigureCopies).toHaveLength(3);
    for(const spec of retiredFigureCopies){
      const row=rows.find(entry=>entry.asset===spec.asset)!;
      expect(()=>validateCuratedSourceAsset(row)).not.toThrow();
      for(const mutation of [{license:'MIT'},{metadata_status:'verified'},
        {rights_status:'source-license-verified'},{sha256:'0'.repeat(64)},
        {source_asset_sha256:'0'.repeat(64)},{publication_status:'active'},
        {used_in:['00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1.md']}])
        expect(()=>validateCuratedSourceAsset({...row,...mutation})).toThrow();
    }
  });

  // This whole-registry integration check hashes source archives and decodes
  // every curated figure. Observed local runs take 24 s in isolation and 36 s
  // after a full build. This is a content-integrity check, not a speed benchmark;
  // keep every assertion and give only this integration case a 60 s budget.
  it('integrates both already-pinned sequence-state objects without weakening source checks',()=>{
    const entries=(parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'),{merge:true}) as {assets:CuratedAsset[]}).assets
      .filter(x=>x.asset.startsWith('00 Учебник/Assets/Figures/curated/sequence-state-2026-11-03/'));
    expect(entries).toHaveLength(2);
    for(const entry of entries){
      expect(()=>validateCuratedSourceAsset(entry)).not.toThrow();
      for(const mutation of [{author:'Unknown'},{sha256:'0'.repeat(64)},
        {source_sha256:'0'.repeat(64)},{source_page:1},{license:'MIT'},
        {license_evidence_sha256:'0'.repeat(64)},{crop_box_px:[0,0,10,10]},
        {dimensions_px:[1,1]}])
        expect(()=>validateCuratedSourceAsset({...entry,...mutation})).toThrow();
    }
  });

  it('rejects altered bytes, source identity, license and crop metadata for PyTorch copy traces',()=>{
    const entries=(parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'),{merge:true}) as {assets:CuratedAsset[]}).assets
      .filter(x=>x.asset.startsWith('00 Учебник/Assets/Figures/curated/gpu-copy-2026-11-04/'));
    expect(entries).toHaveLength(4);
    for(const entry of entries){
      expect(()=>validateCuratedSourceAsset(entry)).not.toThrow();
      for(const mutation of [{author:'Unknown'},{sha256:'0'.repeat(64)},
        {source_asset_sha256:'0'.repeat(64)},{commit:'main'},{license:'MIT'},
        {source_image_url:'https://example.org/trace.png'},
        {license_evidence_sha256:'0'.repeat(64)},{crop_box_px:[0,0,10,10]},
        {dimensions_px:[1,1]},{metadata_status:'unverified'}])
        expect(()=>validateCuratedSourceAsset({...entry,...mutation})).toThrow();
    }
  });

  it('pins the Harvard archive figure inventory without licensing the retired adaptation', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }).assets;
    const historical = JSON.parse(readFileSync(resolve(root, 'docs/audit/2026-11-09-release-readiness.json'), 'utf8'))
      .courseImageReconciliation.pendingHarvardFileAndSHA256 as [string, string][];
    expect(historical).toHaveLength(34);
    for (const [file, hash] of historical) {
      const path = 'Assets/' + file;
      const entries = registry.filter((row: CuratedAsset) => row.asset === path);
      expect(entries).toHaveLength(1);
      const [row] = entries;
      expect(row).toMatchObject({ sha256: hash, source_asset_sha256: hash,
        source_render_commit: 'c1c8e803d61c7c4f96bd1424ec5a0501f20f7730' });
      expect(sha256(resolve(root, path))).toBe(hash);
      expect(row.source_image_url).toContain('/' + row.source_render_commit + '/');
      if (file.endsWith('/gradient-attack.svg')) {
        expect(row).toMatchObject({ license: null, rights_status: 'needs-confirmation', publication_status: 'retired', used_in: [] });
      } else {
        expect(row).toMatchObject({ license: 'CC BY-NC-SA 4.0', rights_status: 'source-license-verified' });
        expect(row.used_in).toHaveLength(2);
      }
    }
    const asset = '00 Учебник/Assets/Figures/curated/harvard-robustness-2026-11-09/fgsm-geometry.svg';
    expect(registry.find((row: CuratedAsset) => row.asset === asset)).toMatchObject({
      sha256: sha256(resolve(root, asset)), license: 'CC BY-NC-ND 4.0', derivation: 'source-svg-copy',
    });
  });

  it('pins four legacy paper crops without treating visual provenance as a reuse license', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: CuratedAsset[];
    };
    expect(legacyPdfFigureEvidence).toHaveLength(4);
    expect(legacyPdfFigureEvidence.filter(({ record }) => record.license === 'CC BY 4.0')).toHaveLength(2);
    const pending = legacyPdfFigureEvidence.filter(({ record }) => record.rights_status === 'needs-confirmation');
    expect(pending).toHaveLength(2);
    for (const { record } of legacyPdfFigureEvidence) {
      const entry = registry.assets.find(asset => asset.asset === record.asset)!;
      expect(() => validateCuratedSourceAsset(entry)).not.toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, sha256: '0'.repeat(64) })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, source_page: 999 })).toThrow();
    }
    for (const { record } of pending) {
      expect(record.license).toBeNull();
      expect(() => validateCuratedSourceAsset({
        ...record, license: 'CC BY 4.0', rights_status: 'source-license-verified',
      })).toThrow();
    }
  });

  it('keeps the three archived VLM originals out of their former reader chapters', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: CuratedAsset[];
    };
    expect(retiredVlmFigures).toHaveLength(3);
    for (const expected of retiredVlmFigures) {
      const row = registry.assets.find(asset => asset.asset === expected.asset)!;
      expect(() => validateCuratedSourceAsset(row)).not.toThrow();
      expect(() => validateCuratedSourceAsset({ ...row, publication_status: 'active' })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...row, sha256: '0'.repeat(64) })).toThrow();
    }
  });

  it('verifies the complete licensed PP-DocLayoutV2 figure in both textbook editions', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: CuratedAsset[];
    };
    const row = registry.assets.find(asset => asset.id === 'curated-pp-doclayout-v2-reading-order-f02cc9356b2a')!;
    expect(() => validateCuratedSourceAsset(row)).not.toThrow();
    for (const changed of [
      { license: 'Apache-2.0' }, { source_image_url: row.source_image_url!.replace('v1/', '/') },
      { sha256: '0'.repeat(64) }, { crop_box_px: [0, 0, 1440, 436] },
    ]) expect(() => validateCuratedSourceAsset({ ...row, ...changed })).toThrow();
  });

  it('pins the Mooncake and Dynamo originals and rejects source, license, and retirement drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: CuratedAsset[];
    };
    for (const original of servingRepositoryOriginals) {
      const row = registry.assets.find(asset => asset.asset === original.asset)!;
      expect(() => validateCuratedSourceAsset(row)).not.toThrow();
      expect(() => validateCuratedSourceAsset({ ...row, commit: '0'.repeat(40) })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...row, license_url: 'https://example.com/LICENSE' })).toThrow();
    }
    const archived = registry.assets.find(asset => asset.asset === servingRepositoryReview.archivedDynamoOverview.asset)!;
    expect(() => validateCuratedSourceAsset({ ...archived, publication_status: 'active' })).toThrow();
  });

  it('pins the original NCCL and Megatron figures and the licenses of the selected revisions', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: CuratedAsset[];
    };
    expect(distributedNvidiaFigures.map(row => row.license)).toEqual([
      'Apache-2.0', 'Apache-2.0', 'Apache-2.0', 'Apache-2.0', 'BSD-3-Clause',
    ]);
    for (const original of distributedNvidiaFigures) {
      const rows = registry.assets.filter(row => row.asset === original.asset);
      expect(rows).toHaveLength(1);
      expect(() => validateCuratedSourceAsset(rows[0])).not.toThrow();
      expect(() => validateCuratedSourceAsset({ ...rows[0], commit: '0'.repeat(40) })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...rows[0], license_url: 'https://example.com/LICENSE' })).toThrow();
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
    const invalid: string[] = [];
    for (const entry of curatedEntries) {
      expect(entry.id).toMatch(/^curated-[a-z0-9-]+-[a-f0-9]{12}$/);
      try {
        validateCuratedSourceAsset(entry);
      } catch (cause) {
        invalid.push(entry.asset + ': ' + (cause instanceof Error ? cause.message : String(cause)));
      }
    }
    expect(invalid).toEqual([]);
  }, 60_000);

  it('rejects source drift for native SVG plots and embedded paper rasters', () => {
    const {assets} = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as {
      assets: CuratedAsset[];
    };
    const ids = [
      'curated-dolma-figure31-c4-f3ebc96ef6ea', 'curated-c4-figure5-pmi-f75800b58acb',
      'curated-peft-qlora-fig2-6690d5f00e17', 'curated-peft-qlora-fig6-cc100be66d62',
      'curated-sft-tulu-figure5-6cda6319b36a', 'curated-posttraining-tulu-figure23-d1aa186a4c2c',
      'curated-scaling-sardana-figure5-2eeafe50cf15',
    ];
    for (const id of ids) {
      const row = assets.find(asset => asset.id === id)!;
      expect(row, id).toBeDefined();
      validateCuratedSourceAsset(row);
      for (const drift of [
        { source_image_url: row.source_image_url!.replace(/v[125]\//, '/').replace('#page=6', '#page=7') },
        { sha256: '0'.repeat(64) },
        { license: 'MIT' },
        { crop_box_px: [0, 0, 10, 10] },
      ]) expect(() => validateCuratedSourceAsset({...row, ...drift})).toThrow();
    }
    const sardana = assets.find(asset => asset.id === ids.at(-1))!;
    expect(() => validateCuratedSourceAsset({...sardana, source_pdf_object: 33})).toThrow();
    const dolma = assets.find(asset => asset.id === ids[0])!;
    expect(() => validateCuratedSourceAsset({...dolma, source_viewbox: [0, 0, 100, 100]})).toThrow();
  });

  it('binds versioned arXiv originals to their publisher asset and license evidence', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')) as { assets: CuratedAsset[] };
    const row = registry.assets.find(entry => entry.id === 'curated-distserve-arxiv-teaser-bae5d6fba7ac')!;
    expect(() => validateCuratedSourceAsset(row)).not.toThrow();
    for (const mutation of [
      { source_image_url: 'https://arxiv.org/html/2401.09670v2/teaser.svg' },
      { source_url: 'https://arxiv.org/html/2401.09670#S1.F1' },
      { license_evidence: 'https://arxiv.org/html/2401.09670v2' },
      { license: 'MIT' },
      { sha256: '0'.repeat(64) },
      { crop_box_px: [0, 0, 40, 40] },
    ]) expect(() => validateCuratedSourceAsset({ ...row, ...mutation })).toThrow();
  });
  it('validates the complete original BERTScore image and rejects altered source/license metadata', () => {
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'),{merge:true}) as {assets:CuratedAsset[]};
    const entry=registry.assets.find(row=>row.id==='curated-bertscore-recall-d917184b34ff')!;
    expect(()=>validateCuratedSourceAsset(entry)).not.toThrow();
    for(const mutation of [{license:'unverified'},{sha256:'0'.repeat(64)},
      {source_asset_sha256:'0'.repeat(64)},{license_evidence_sha256:'0'.repeat(64)},
      {source_image_url:entry.source_image_url!.replace('19e7f551fe4fa43fdd07b8129ae947015b902b2d','master')}])
      expect(()=>validateCuratedSourceAsset({...entry,...mutation})).toThrow();
  });
  it('validates the pinned PEFT original raster packet and rejects altered provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const row = registry.assets.find(entry => entry.id === 'curated-peft-20261029-398088f0cf9b')!;
    expect(() => validateCuratedSourceAsset(row)).not.toThrow();
    for (const changed of [{ commit: 'main' }, { license: 'unverified' },
      { license_evidence_sha256: '0'.repeat(64) }, { sha256: '0'.repeat(64) },
      { source_asset: row.source_asset!.replace('lora-animated.gif', '../lora-animated.gif') }])
      expect(() => validateCuratedSourceAsset({ ...row, ...changed })).toThrow();
  });

  it('checks both original small-data figures and rejects changed image/license provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    for (const id of ['curated-small-data-asapp-a4a3f45c72dc', 'curated-small-data-mosbach-a994bb0e2860']) {
      const row = registry.assets.find(entry => entry.id === id)!;
      expect(() => validateCuratedSourceAsset(row)).not.toThrow();
      for (const changed of [{ commit: 'master' }, { license: 'unverified' },
        { source_asset_sha256: '0'.repeat(64) }, { license_evidence_sha256: '0'.repeat(64) },
        { source_asset: row.source_asset!.replace(/[^/]+$/, '../figure.png') }])
        expect(() => validateCuratedSourceAsset({ ...row, ...changed })).toThrow();
    }
  });

  it('validates the retained NeMo JPEG and exact-size PNG conversion without weakening source checks', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const row = registry.assets.find(entry => entry.id === 'curated-arithmetic-nemo-source-exact-1f445c962396')!;
    expect(() => validateCuratedSourceAsset(row)).not.toThrow();
    expect(pngDimensions(resolve(root, row.asset))).toEqual([1768, 604]);
    expect(() => validateCuratedSourceAsset({ ...row, source_asset_sha256: '0'.repeat(64) })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...row, sha256: '0'.repeat(64) })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...row, source_asset: row.source_asset!.replace('/arithmetic-review-2026-09/', '/../') })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...row, provenance_confirmation: 'user-confirmed-open-materials' })).toThrow();
  });

  it('bundles the original TE scaling SVG and CSS without changing the diagram', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const row = registry.assets.find(entry => entry.id === 'curated-precision-te-current-delayed-a40da6563f7f')!;
    expect(() => validateCuratedSourceAsset(row)).not.toThrow();
    for (const edit of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
      { source_stylesheet_sha256: '0'.repeat(64) }, { commit: 'main' }, { license: 'MIT' }]) {
      expect(() => validateCuratedSourceAsset({ ...row, ...edit })).toThrow();
    }
  });

  it('retains the complete MXFP8 row/column source diagram and rejects altered provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const row = registry.assets.find(entry => entry.id === 'curated-precision-te-row-column-603a59f25341')!;
    expect(() => validateCuratedSourceAsset(row)).not.toThrow();
    for (const edit of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
      { source_stylesheet_sha256: '0'.repeat(64) }, { commit: 'main' }, { license: 'MIT' },
      { source_asset: 'publishing/fixtures/../mxfp8_row_col.svg' },
      { source_url: row.source_url!.replace('mxfp8_row_col.svg', 'unreviewed.svg') },
      { asset: row.asset.replace('te-mxfp8-row-column.svg', 'unreviewed.svg') }]) {
      expect(() => validateCuratedSourceAsset({ ...row, ...edit })).toThrow();
    }
    const svg = readFileSync(resolve(root, row.asset), 'utf8');
    expect(svg).toContain('viewBox="0 0 650 450"');
    expect(svg).toContain('Rowwise (1x32 blocks)');
    expect(svg).toContain('Columnwise (32x1 blocks)');
  });

  it('accepts archived viewport-only SVG crops but rejects content edits and invalid bounds', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    // Crops with embedded upstream CSS have a different derivation contract.
    // This check covers viewport-only derivatives with a retained original SVG.
    const entries = registry.assets.filter(a => a.derivation === 'source-svg-crop' && a.source_asset);
    expect(entries.length).toBeGreaterThan(0);
    for (const entry of entries) {
      expect(() => validateCuratedSourceAsset(entry)).not.toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, source_asset_sha256: '0'.repeat(64) })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...entry, source_url: entry.source_url!.replace(entry.commit!, 'main') })).toThrow();
      const original = readFileSync(resolve(root, entry.source_asset!), 'utf8');
      const cropped = readFileSync(resolve(root, entry.asset), 'utf8');
      const full = original.match(/viewBox="([^"]+)"/)![1]!.trim().split(/\s+/).map(Number);
      const outsideBounds = cropped.replace(/viewBox="[^"]+"/,
        `viewBox="${full[0]} ${full[1]} ${full[2]! + 1} ${full[3]! + 1}"`);
      const editedContent = cropped.replace('</svg>', '<path d="M0 0 L1 1"/></svg>');
      expect(outsideBounds).not.toBe(cropped);
      expect(editedContent).not.toBe(cropped);
      expect(() => expectViewportOnlySvgCrop(original, outsideBounds)).toThrow();
      expect(() => expectViewportOnlySvgCrop(original, editedContent)).toThrow();
    }
  });

  it('accepts nested arXiv figure paths but rejects unpinned or unrelated sources', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: CuratedAsset[];
    };
    const valid = registry.assets.find(({ asset }) => asset.endsWith('/tulu-ifeval-kl-and-generalization.png'));
    expect(valid).toBeTruthy();
    if (!valid) return;
    expect(() => validateCuratedSourceAsset(valid)).not.toThrow();
    for (const source_image_url of [
      'https://arxiv.org/html/2411.15124/figures_images/post-training/rl/plot_2_ifeval.png',
      'https://arxiv.org/html/2411.15124v4/figures_images/post-training/rl/plot_2_ifeval.png',
      'https://arxiv.org/html/2411.15124v5/../other.png',
      'https://arxiv.org/html/2411.15124v5//other.png',
      'https://example.org/html/2411.15124v5/figure.png',
    ]) {
      expect(() => validateCuratedSourceAsset({ ...valid, source_image_url })).toThrow();
    }
    expect(() => validateCuratedSourceAsset({ ...valid, source_asset_sha256: '0'.repeat(64) })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...valid, derivation: 'source-svg-copy' })).toThrow();
    expect(() => validateCuratedSourceAsset({ ...valid, source_image_url: valid.source_image_url?.replace(/\.png$/, '.svg') })).toThrow();
  });

  it.each(['curated-sequence-rwkv4-92f374fb9a52', 'curated-sequence-rwkv7-046abe9e2b39'])(
    'accepts an unchanged version-pinned arXiv SVG: %s', (id) => {
      const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
        assets: CuratedAsset[];
      };
      const valid = registry.assets.find(asset => asset.id === id)!;
      expect(valid).toBeTruthy();
      expect(() => validateCuratedSourceAsset(valid)).not.toThrow();
      expectDecodableImage(resolve(root, valid.asset));
      for (const invalid of [
        { derivation: 'source-raster-copy' },
        { derivation: 'redrawn' },
        { asset: valid.asset.replace(/\.svg$/, '.png') },
        { source_image_url: valid.source_image_url?.replace(/\.svg$/, '.png') },
        { source_url: valid.source_url?.replace(/v\d+$/, '') },
        { source_image_url: valid.source_image_url?.replace(/v\d+\//, 'v99/') },
        { source_asset_sha256: '0'.repeat(64) },
        { sha256: '0'.repeat(64) },
        { source_asset_sha256: '0'.repeat(64), sha256: '0'.repeat(64) },
        { source_asset_sha256: undefined },
        { author: '' },
        { license: '' },
        { license_url: undefined },
        { metadata_status: 'unverified' },
        { modifications: '' }
      ]) {
        expect(() => validateCuratedSourceAsset({ ...valid, ...invalid })).toThrow();
      }
    }
  );

  it('retains exact reviewed distributed-system figures and public license notices', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: (CuratedAsset & { license_copy?: string; public_license_copy?: string })[];
    };
    const reviewed = registry.assets.filter(entry => /\/(fsdp|expert-hybrid|fault-tolerance|network-checkpoint)-review-2026-10\//.test(entry.asset));
    expect(reviewed).toHaveLength(10);
    for (const entry of reviewed) {
      expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(sha256(resolve(root, entry.asset))).toBe(entry.sha256);
      expectDecodableImage(resolve(root, entry.asset));
      validateCuratedSourceAsset(entry);
      if (entry.license === 'MIT' || entry.license === 'BSD-3-Clause') {
        expect(entry.license_copy).toBeTruthy();
        expect(entry.public_license_copy).toMatch(/^site\/public\/licenses\//);
        expect(readFileSync(resolve(root, entry.public_license_copy!))).toEqual(
          readFileSync(resolve(root, entry.license_copy!))
        );
      }
    }
  });

  it('preserves the five reviewed original PagedAttention figures and their named license', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: (CuratedAsset & { rights_verification?: string })[];
    };
    const paths = [
      'topics-53-59-source-first/paged-attention-fragmentation.png',
      'topics-53-59-source-first/paged-attention-blocks.png',
      'topics-53-59-source-first/paged-attention-logical.png',
      'stanford-cs336-2026/inference-evaluation/paged-attention-sharing.png',
      'stanford-cs336-2026/inference-evaluation/paged-attention-copy-on-write.png'
    ];
    const chapter = readFileSync(resolve(root, '00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md'), 'utf8');
    for (const path of paths) {
      const entry = registry.assets.find(e => e.asset === `00 Учебник/Assets/Figures/curated/${path}`);
      expect(entry).toBeDefined();
      expect(entry!.license).toBe('CC BY 4.0');
      expect(entry!.license_url).toBe('https://creativecommons.org/licenses/by/4.0/');
      expect(entry!.rights_verification).toContain('2309.06180v1');
      expect(sha256(resolve(root, entry!.asset))).toBe(entry!.sha256);
      expectDecodableImage(resolve(root, entry!.asset));
      expect(chapter).toContain(path);
    }
    expect(chapter).not.toContain('harvard/inference/kv-cache-fragmentation.svg');
  });

  it('decodes the original indexed HF KV illustration without recompressing it', () => {
    const asset = '00 Учебник/Assets/Figures/curated/edls-neural-kv-2026-10-26/hf-kv-cache.png';
    const path = resolve(root, asset);
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'),
      { merge: true }) as { assets: CuratedAsset[] };
    const entry = registry.assets.find(row => row.asset === asset);
    expect(entry).toBeDefined();
    validateCuratedSourceAsset(entry!);
    expect(entry!.used_in).toEqual([
      '06 Практика/14 Собрать минимальный inference engine.md',
      '00 Учебник/_index.md',
      'en/00 Textbook/_index.md',
      '00 Учебник/07 Анатомия современной LLM/02 Современный decoder block.md',
    ]);
    for (const owner of entry!.used_in!)
      expect(readFileSync(resolve(root, owner), 'utf8')).toContain('![[02 Areas/ML & DL/' + asset);
    const bytes = readFileSync(path);
    expect(bytes[25]).toBe(3); // indexed-colour PNG, not a rewritten RGB copy
    expect(validateRenderedPngBytes(bytes)).toEqual([1348, 466]);
    const chunks = [bytes.subarray(0, 8)];
    for (let offset = 8; offset < bytes.length;) {
      const end = offset + 12 + bytes.readUInt32BE(offset);
      if (bytes.toString('ascii', offset + 4, offset + 8) !== 'PLTE')
        chunks.push(bytes.subarray(offset, end));
      offset = end;
    }
    expect(() => validateRenderedPngBytes(Buffer.concat(chunks))).toThrow();
    expect(() => validateRenderedPngBytes(bytes.subarray(0, -1))).toThrow();
    const corrupt = Buffer.from(bytes);
    corrupt[Math.floor(corrupt.length / 2)] ^= 1;
    expect(() => validateRenderedPngBytes(corrupt)).toThrow();
  });

  it('pins the two original HF scheduling figures to the image dataset license and bytes', () => {
    const registry = parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'),{merge:true});
    const rows = registry.assets.filter((entry:CuratedAsset)=>entry.asset.includes('/scheduling-2026-10-25/'));
    expect(rows).toHaveLength(2);
    const chapter = readFileSync(resolve(root,
      '00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching.md'),'utf8');
    const practicePath = '06 Практика/14 Собрать минимальный inference engine.md';
    const practice = readFileSync(resolve(root, practicePath), 'utf8');
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      expect(row.used_in).toEqual([
        '00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching.md',
        practicePath,
        ...(row.asset.endsWith('/hf-continuous-batching.png')
          ? ['06 Практика/14b Планировщик запросов и KV-память.md'] : [])
      ]);
      expect(chapter).toContain(row.asset);
      expect(practice).toContain(row.asset);
      if (row.asset.endsWith('/hf-continuous-batching.png'))
        expect(readFileSync(resolve(root, '06 Практика/14b Планировщик запросов и KV-память.md'), 'utf8')).toContain(row.asset);
      for (const invalid of [{sha256:'0'.repeat(64)}, {commit:'0'.repeat(40)},
        {source_image_url:row.source_image_url.replace(row.commit,'main')},
        {source_asset_sha256:'0'.repeat(64)}, {license:'Apache-2.0'},
        {license_evidence_sha256:'0'.repeat(64)}, {crop_box_px:[0,0,10,10]}])
        expect(()=>validateCuratedSourceAsset({...row,...invalid})).toThrow();
    }
    expect(chapter).toContain('CC BY-NC-SA 4.0');
    expect(chapter).not.toContain('/hf-ragged-batching.png');
  });

  it('preserves the six original scheduling and serving illustrations with public license copies', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: (CuratedAsset & { license_copy?: string; public_license_copy?: string; used_in: string[] })[];
    };
    const reviewed = registry.assets.filter(entry => /\/(inference-scheduling|serving-engines)-review-2026-10\//.test(entry.asset));
    expect(reviewed).toHaveLength(6);
    for (const entry of reviewed) {
      expect(entry.sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(sha256(resolve(root, entry.asset))).toBe(entry.sha256);
      expectDecodableImage(resolve(root, entry.asset));
      validateCuratedSourceAsset(entry);
      expect(['Apache-2.0', 'MIT']).toContain(entry.license);
      expect(entry.license_copy).toBeTruthy();
      expect(entry.public_license_copy).toMatch(/^site\/public\/licenses\//);
      expect(readFileSync(resolve(root, entry.public_license_copy!))).toEqual(
        readFileSync(resolve(root, entry.license_copy!))
      );
      const studyUse = vllmSourceLibrary.figures.find(figure => figure.asset === entry.asset);
      if (studyUse) expect(entry.used_in).toEqual(studyUse.used_in);
      else expect(entry.used_in).toHaveLength(1);
      for (const owner of entry.used_in) {
        expect(readFileSync(resolve(root, owner), 'utf8')).toContain(entry.asset);
      }
    }
  });

  it('preserves the four additional serving source figures with exact hashes and attribution', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: (CuratedAsset & { license_copy?: string; public_license_copy?: string; used_in: string[] })[];
    };
    const reviewed = registry.assets.filter(entry => entry.asset.includes('/serving-originals-2026-10/'));
    expect(reviewed).toHaveLength(4);
    for (const entry of reviewed) {
      validateCuratedSourceAsset(entry);
      expect(sha256(resolve(root, entry.asset))).toBe(entry.sha256);
      expectDecodableImage(resolve(root, entry.asset));
      expect(entry.license_copy).toBeTruthy();
      expect(readFileSync(resolve(root, entry.public_license_copy!)).equals(readFileSync(resolve(root, entry.license_copy!)))).toBe(true);
      expect(entry.used_in).toHaveLength(1);
      expect(readFileSync(resolve(root, entry.used_in[0]), 'utf8')).toContain(entry.asset);
    }
  });

  it('preserves six positional source figures and their exact copying/crop provenance', () => {
    const registry = parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'),{merge:true}) as {assets:CuratedAsset[]};
    const rows = registry.assets.filter(row => Object.hasOwn(positionalFigures,row.asset));
    expect(rows).toHaveLength(6);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const invalid of [
        {author:'Unknown'}, {source_url:'https://example.org/'}, {license:'public-domain'},
        {source_asset:'elsewhere/a.pdf'}, {sha256:'0'.repeat(64)},
        {source_asset_sha256:'0'.repeat(64)}, {source_page_sha256:'0'.repeat(64)},
        {license_evidence_sha256:'0'.repeat(64)}, {derivation:'redraw'},
        {provenance_confirmation:'unverified'}, {metadata_status:'unverified'}
      ]) expect(() => validateCuratedSourceAsset({...row,...invalid})).toThrow();
      if (row.derivation === 'pdf-page-render-crop')
        for (const invalid of [{source_page:1},{render_dpi:72},{crop_box_px:[0,0,1,1]}])
          expect(() => validateCuratedSourceAsset({...row,...invalid})).toThrow();
      if (row.commit)
        expect(() => validateCuratedSourceAsset({...row,commit:'0'.repeat(40)})).toThrow();
    }
  });

  it('preserves five masking/head originals and distinguishes unverified legacy permissions', () => {
    const registry = parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'),{merge:true}) as {assets:CuratedAsset[]};
    const rows = registry.assets.filter(row => Object.hasOwn(maskingHeadsFigures,row.asset));
    expect(rows).toHaveLength(5);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const invalid of [
        {author:'Unknown'}, {license:'public-domain'}, {source_image_url:'https://example.org/a.png'},
        {source_asset:'elsewhere/a.png'}, {sha256:'0'.repeat(64)},
        {source_asset_sha256:'0'.repeat(64)}, {source_page_sha256:'0'.repeat(64)},
        {derivation:'redraw'}, {provenance_confirmation:'unverified'}
      ]) expect(() => validateCuratedSourceAsset({...row,...invalid})).toThrow();
      if (row.metadata_status === 'needs-confirmation') {
        expect(() => validateCuratedSourceAsset({...row,metadata_status:'verified'})).toThrow();
        expect(() => validateCuratedSourceAsset({...row,rights_status:'open-license-verified'})).toThrow();
      } else {
        expect(() => validateCuratedSourceAsset({...row,license_evidence_sha256:'0'.repeat(64)})).toThrow();
      }
    }
  });

  it('preserves all four self-attention source figures and rejects provenance drift', () => {
    const registry = parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'),{merge:true}) as {assets:CuratedAsset[]};
    const rows = registry.assets.filter(row => Object.hasOwn(selfAttentionFigures,row.asset));
    expect(rows).toHaveLength(4);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const invalid of [
        {author:'Unknown'}, {license:'public-domain'}, {source_image_url:'https://example.org/a.png'},
        {sha256:'0'.repeat(64)}, {source_asset_sha256:'0'.repeat(64)},
        {source_page_sha256:'0'.repeat(64)}, {license_evidence_sha256:'0'.repeat(64)},
        {derivation:'redraw'}, {provenance_confirmation:'unverified'}
      ]) expect(() => validateCuratedSourceAsset({...row,...invalid})).toThrow();
    }
  });

  it('preserves recurrent-attention figures, licensed crops and honest legacy permissions', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const rows = registry.assets.filter(row => Object.hasOwn(attentionTranslationFigures, row.asset));
    expect(rows).toHaveLength(7);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const invalid of [
        { source_url: 'https://example.org/' }, { source_image_url: 'https://example.org/a.svg' },
        { source_asset_sha256: '0'.repeat(64) }, { sha256: '0'.repeat(64) },
        { source_page_sha256: '0'.repeat(64) }, { author: 'Unknown' },
        { license: 'public-domain' }, { derivation: 'redraw' },
        { provenance_confirmation: 'unverified' }
      ]) expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
      if (row.derivation === 'pdf-page-render-crop') {
        for (const invalid of [{ source_page: 1 }, { render_dpi: 96 }, { crop_box_px: [0,0,10,10] },
          { license_evidence_sha256: '0'.repeat(64) }])
          expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
      }
      if (row.metadata_status === 'needs-confirmation') {
        expect(() => validateCuratedSourceAsset({ ...row, metadata_status: 'verified' })).toThrow();
        expect(() => validateCuratedSourceAsset({ ...row, rights_status: 'open-license-verified' })).toThrow();
      }
    }
  });

  it('pins the seq2seq encoder interface original and rejects changed provenance',()=>{
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')) as {assets:CuratedAsset[]};
    const entry=registry.assets.find(x=>x.asset==='00 Учебник/Assets/Figures/curated/seq2seq-bridge-2026-11-03/d2l-encoder-decoder.svg')!;
    expect(()=>validateCuratedSourceAsset(entry)).not.toThrow();
    for(const invalid of [{author:'Unknown'},{commit:'main'},{license:'CC0'},
      {sha256:'0'.repeat(64)},{source_asset_sha256:'0'.repeat(64)},
      {source_page_sha256:'0'.repeat(64)},{license_evidence_sha256:'0'.repeat(64)},
      {source_image_url:'https://example.org/a.svg'},{derivation:'source-raster-copy'}])
      expect(()=>validateCuratedSourceAsset({...entry,...invalid})).toThrow();
  });

  it('preserves all seven seq2seq source figures and rejects provenance drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const rows = registry.assets.filter(row => Object.hasOwn(seq2seqFigures, row.asset));
    expect(rows).toHaveLength(7);
    for (const row of rows) {
      validateCuratedSourceAsset(row);
      for (const invalid of [
        { source_url: 'https://example.org/' }, { source_image_url: 'https://example.org/image.svg' },
        { author: 'Unknown' }, { commit: '0'.repeat(40) }, { license: 'MIT' },
        { source_asset: 'missing/source.svg' }, { source_asset_sha256: '0'.repeat(64) },
        { sha256: '0'.repeat(64) }, { source_page_sha256: '0'.repeat(64) },
        { license_evidence_sha256: '0'.repeat(64) }, { derivation: 'redraw' },
        { provenance_confirmation: 'unchecked' }, { metadata_status: 'needs-confirmation' }
      ]) expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });

  it('rejects mismatched arXiv source-archive provenance', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entry = registry.assets.find(e => e.provenance_confirmation === 'pinned-arxiv-source-archive-member')!;
    expect(entry).toBeDefined();
    validateCuratedSourceAsset(entry);
    for (const invalid of [
      { source_url: 'https://arxiv.org/src/2407.08608' },
      { source_member: '../pingpong_pipelining.png' },
      { source_asset_sha256: '0'.repeat(64) },
      { source_member_sha256: '0'.repeat(64) },
      { sha256: '0'.repeat(64) },
      { derivation: 'source-raster-copy' },
    ]) expect(() => validateCuratedSourceAsset({ ...entry, ...invalid })).toThrow();
  });

  it('preserves vision-encoder plots and rejects source or license drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const figures = registry.assets.filter(row => row.asset.includes('/vision-encoders-review-2026-10/'));
    expect(figures).toHaveLength(3);
    for (const row of figures) {
      validateCuratedSourceAsset(row);
      for (const invalid of [
        { author: 'Unknown' }, { source_page: 1 }, { crop_box_px: [0, 0, 1, 1] },
        { source_url: 'https://example.org/' }, { license: 'MIT' },
        { source_asset_sha256: '0'.repeat(64) }, { sha256: '0'.repeat(64) }
      ]) expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });

  it('preserves BLIP-2 architecture, mask and stage-two panels and rejects provenance drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const figures = registry.assets.filter(row => row.asset.includes('/connectors-review-2026-10/'));
    expect(figures).toHaveLength(3);
    for (const row of figures) {
      validateCuratedSourceAsset(row);
      for (const invalid of [
        { author: 'Unknown' }, { source_page: 99 }, { crop_box_px: [0, 0, 1, 1] },
        { source_url: 'https://example.org/' }, { license: 'MIT' },
        { source_asset_sha256: '0'.repeat(64) }, { sha256: '0'.repeat(64) }
      ]) expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });

  it('preserves VLM evaluation and EPD figures and rejects source/rights drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const figures = registry.assets.filter(row => row.asset.includes('/vlm-evaluation-review-2026-10/'));
    expect(figures).toHaveLength(5);
    for (const row of figures) {
      validateCuratedSourceAsset(row);
      for (const invalid of [
        { author: 'Unknown' }, { source_page: 99 }, { crop_box_px: [0, 0, 1, 1] },
        { source_url: 'https://example.org/' }, { license: 'public-domain' },
        { source_asset_sha256: '0'.repeat(64) }, { sha256: '0'.repeat(64) },
        { license_evidence_sha256: '0'.repeat(64) }
      ]) expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });

  it('keeps repaired CS231n and historical Molmo figures tied to their exact PDF crops', () => {
    const archive = '05 Источники/Originals/vlm-evaluation-serving-2026-10-20/';
    const refs = JSON.parse(readFileSync(resolve(root, archive + 'references.json'), 'utf8')) as {
      repairs: Array<CuratedAsset & { source_raster: string; source_raster_sha256: string;
        width: number; height: number; source_raster_size_px: number[] }>;
    };
    expect(refs.repairs).toHaveLength(2);
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    for (const repair of refs.repairs) {
      const row = registry.assets.find(entry => entry.asset === repair.asset)!;
      expect(row.sha256).toBe(repair.sha256);
      validateCuratedSourceAsset(row);
      if (row.asset.endsWith('/molmo-figure11-openness.png')) {
        expect(row.source_asset).toBe(repair.source_asset);
        expect(row.source_asset_sha256).toBe(repair.source_asset_sha256);
        expect(row.former_used_in).toHaveLength(2);
        for (const owner of row.former_used_in!) {
          const chapter = readFileSync(resolve(root, owner), 'utf8');
          expect(chapter).toContain(row.figure_url);
          expect(chapter).not.toContain(row.asset);
        }
      } else {
        expect(row.source_asset).toMatch(/^raw\/papers\/external\/vlm-2026\//);
      }
      expect(row.modifications).toContain(archive + 'figure-repairs.json');
      for (const [file, hash] of [[repair.asset, repair.sha256],
        [repair.source_asset, repair.source_asset_sha256], [repair.source_page_snapshot, repair.source_page_sha256],
        [repair.source_raster, repair.source_raster_sha256]])
        expect(sha256(resolve(root, file!))).toBe(hash);
      expect(pngDimensions(resolve(root, repair.asset))).toEqual([repair.width, repair.height]);
      expect(pngDimensions(resolve(root, repair.source_raster))).toEqual(repair.source_raster_size_px);
    }
  });

  it('preserves omni-source figures and rejects license, crop and source drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const figures = registry.assets.filter(row => row.asset.includes('/omni-review-2026-10/'));
    expect(figures).toHaveLength(6);
    for (const row of figures) {
      validateCuratedSourceAsset(row);
      for (const invalid of [
        { author: 'Unknown' }, { source_page: 99 }, { crop_box_px: [0, 0, 1, 1] },
        { source_url: 'https://example.org/' }, { license: 'MIT' },
        { source_asset_sha256: '0'.repeat(64) }, { sha256: '0'.repeat(64) },
        { license_evidence_sha256: '0'.repeat(64) }
      ]) expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });

  it('keeps the complete TMRoPE figure tied to its before image and exact PDF crop', () => {
    const archive = '05 Источники/Originals/omni-2026-10-20/';
    const refs = JSON.parse(readFileSync(resolve(root, archive + 'references.json'), 'utf8')) as {
      repairs: Array<CuratedAsset & { source_raster: string; source_raster_sha256: string }>;
    };
    expect(refs.repairs).toHaveLength(1);
    const repair = refs.repairs[0]!;
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const row = registry.assets.find(entry => entry.asset === repair.asset)!;
    expect(row.sha256).toBe(repair.sha256);
    validateCuratedSourceAsset(row);
    expect(row.source_asset).toBe(repair.source_asset);
    expect(row.source_asset_sha256).toBe(repair.source_asset_sha256);
    expect(row.source_page_snapshot).toBe(repair.source_page_snapshot);
    expect(row.source_page_sha256).toBe(repair.source_page_sha256);
    expect(row.modifications).toContain(archive + 'figure-repairs.json');
    for (const [file, hash] of [[repair.asset, repair.sha256],
      [repair.source_asset, repair.source_asset_sha256], [repair.source_page_snapshot, repair.source_page_sha256],
      [repair.source_raster, repair.source_raster_sha256]])
      expect(sha256(resolve(root, file!))).toBe(hash);
    expect(pngDimensions(resolve(root, repair.asset))).toEqual([1745, 475]);
    expect(pngDimensions(resolve(root, repair.source_raster))).toEqual([1423, 290]);
  });

  it('pins the archived Pix2Struct composite without publishing third-party screenshots', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const row = registry.assets.find(entry => entry.id === 'curated-vlm2026-pix2struct-figure1-30034161ebd2')!;
    validateCuratedSourceAsset(row);
    expect(row.source_asset).toBe('05 Источники/Originals/documents-2026-10-20/pix2struct.pdf');
    expect(row.source_asset_sha256).toBe('b899b3beba7723f60cf55f9715e9e02cf409b7ecd3a648156c238bf09f0b58a4');
    expect(row.former_used_in).toHaveLength(2);
    for (const owner of row.former_used_in!) {
      const chapter = readFileSync(resolve(root, owner), 'utf8');
      expect(chapter).toContain(row.figure_url);
      expect(chapter).not.toContain(row.asset);
    }
  });

  it('preserves document-source figures and rejects license, crop and pin drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const figures = registry.assets.filter(row => row.asset.includes('/documents-review-2026-10/'));
    expect(figures).toHaveLength(4);
    for (const row of figures) {
      validateCuratedSourceAsset(row);
      expect(() => validateCuratedSourceAsset({ ...row, license: 'public-domain' })).toThrow();
      expect(() => validateCuratedSourceAsset({ ...row, sha256: '0'.repeat(64) })).toThrow();
      if (row.derivation === 'pdf-page-render-crop')
        expect(() => validateCuratedSourceAsset({ ...row, crop_box_px: [0, 0, 1, 1] })).toThrow();
      else
        expect(() => validateCuratedSourceAsset({ ...row, commit: '0'.repeat(40) })).toThrow();
    }
  });


  it('keeps both retired VLM training rasters tied to their archived PDF crops', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const prefix = '05 Источники/Originals/vlm-training-2026-10-20/';
    const repairs = JSON.parse(readFileSync(resolve(root, prefix + 'figure-repairs.json'), 'utf8')) as Array<{
      asset: string; sha256: string; source_asset: string; source_asset_sha256: string;
      source_raster: string; source_raster_sha256: string;
      source_page_snapshot: string; source_page_sha256: string;
      render_page_size_px: number[]; crop_box_px: number[];
    }>;
    expect(repairs).toHaveLength(2);
    for (const repair of repairs) {
      const entry = registry.assets.find(row => row.asset === repair.asset);
      expect(entry).toBeDefined();
      expect(entry!.sha256).toBe(repair.sha256);
      const figureUrl = repair.asset.endsWith('/internvl25-figure4-training-pipeline.png')
        ? 'https://arxiv.org/pdf/2412.05271v1#page=6'
        : 'https://arxiv.org/pdf/2409.17146v2#page=2';
      expect(entry).toMatchObject({
        publication_status: 'retired', rights_status: 'needs-confirmation',
        used_in: [], license: null, license_url: null,
        source_asset: repair.source_asset, source_asset_sha256: repair.source_asset_sha256,
        figure_url: figureUrl,
      });
      expect(entry!.former_used_in).toHaveLength(2);
      for (const owner of entry!.former_used_in!) {
        const chapter = readFileSync(resolve(root, owner), 'utf8');
        expect(chapter).toContain(entry!.figure_url);
        expect(chapter).not.toContain(repair.asset);
      }
      expect(entry!.modifications).toContain(prefix + 'figure-repairs.json');
      for (const [path, hash] of [
        [repair.asset, repair.sha256], [repair.source_raster, repair.source_raster_sha256],
        [repair.source_asset, repair.source_asset_sha256],
        [repair.source_page_snapshot, repair.source_page_sha256]
      ]) expect(sha256(resolve(root, path!))).toBe(hash);
      expect(pngDimensions(resolve(root, repair.source_page_snapshot))).toEqual(repair.render_page_size_px);
      const [x1, y1, x2, y2] = repair.crop_box_px;
      expect(pngDimensions(resolve(root, repair.asset))).toEqual([x2!-x1!, y2!-y1!]);
      expectDecodableImage(resolve(root, repair.asset));
    }
  });

  it.each(['curated-internvl35-training-5234140e2125', 'curated-molmopoint-guisyn-dcb476acd4a1',
    'curated-mm-detect-figure1-26bb5ad1fbe1', 'curated-pheye-dense-cross-attention-f74aaed3c0c5'])('pins the original CC BY VLM figure %s in both locales', id => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const entry = registry.assets.find(row => row.id === id);
    expect(entry).toBeDefined();
    validateCuratedSourceAsset(entry!);
    expect(() => validateCuratedSourceAsset({...entry!, license: 'public-domain'})).toThrow();
    expect(() => validateCuratedSourceAsset({...entry!, sha256: '0'.repeat(64)})).toThrow();
  });

  it('keeps the two Flamingo copies and Molmo tokenization figure archived and linked', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    for (const [id, formerUseCount] of [
      ['curated-vlm2026-flamingo-figure3-764e83c6d558', 2],
      ['curated-atlas-multimodal-000000000089', 1],
      ['curated-vlm2026-molmo-figure5-35fde220b09c', 3],
    ] as const) {
      const row = registry.assets.find(entry => entry.id === id)!;
      validateCuratedSourceAsset(row);
      expect(row.former_used_in).toHaveLength(formerUseCount);
      for (const owner of row.former_used_in!) {
        const chapter = readFileSync(resolve(root, owner), 'utf8');
        expect(chapter).toContain(row.figure_url);
        expect(chapter).not.toContain(row.asset);
      }
    }
  });

  it('preserves the original Pixtral packing figure and rejects provenance drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const figures = registry.assets.filter(row => row.asset.includes('/resolution-review-2026-10/'));
    expect(figures).toHaveLength(1);
    for (const row of figures) {
      validateCuratedSourceAsset(row);
      for (const invalid of [
        { author: 'Unknown' }, { source_page: 99 }, { crop_box_px: [0, 0, 1, 1] },
        { source_url: 'https://example.org/' }, { license: 'MIT' },
        { source_asset_sha256: '0'.repeat(64) }, { sha256: '0'.repeat(64) },
        { license_evidence_sha256: '0'.repeat(64) }
      ]) expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });

  it('preserves RAG figure panels and rejects provenance or license drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const figures = registry.assets.filter(row => row.asset.includes('/rag-review-2026-10/'));
    expect(figures).toHaveLength(5);
    for (const row of figures) {
      validateCuratedSourceAsset(row);
      for (const invalid of [
        { author: 'Unknown' }, { source_page: 99 }, { crop_box_px: [0, 0, 1, 1] },
        { source_url: 'https://example.org/' }, { source_asset_sha256: '0'.repeat(64) },
        { license: 'MIT' }, { license_evidence_sha256: '0'.repeat(64) },
        { sha256: '0'.repeat(64) }
      ]) expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });

  it('preserves the four agent-security source figures and rejects provenance drift', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as { assets: CuratedAsset[] };
    const figures = registry.assets.filter(row => row.asset.includes('/agent-security-2026-10-21/'));
    expect(figures).toHaveLength(4);
    for (const row of figures) {
      validateCuratedSourceAsset(row);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { author: '' }, { license: 'public domain' }, { source_page: 99 },
        { crop_box_px: [0, 0, 1, 1] }, { source_url: 'https://example.org/' }])
        expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
    }
  });

  it('checks complete Mamba-2 figures and rejects source/crop/license drift',()=>{
    const registry=parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8'),{merge:true}) as {assets:CuratedAsset[]};
    const figures=registry.assets.filter(row=>row.asset.includes('/sequence-alternatives-2026-10-23/'));
    expect(figures).toHaveLength(2);
    for(const row of figures){
      validateCuratedSourceAsset(row);
      for(const invalid of [{source_page:1},{render_dpi:72},{crop_box_px:[0,0,2,2]},
        {license:'MIT'},{author:''},{source_asset_sha256:'0'.repeat(64)},
        {sha256:'0'.repeat(64)},{license_evidence_sha256:'0'.repeat(64)},
        {source_url:'https://example.org/'}])
        expect(()=>validateCuratedSourceAsset({...row,...invalid})).toThrow();
    }
  });
  it('keeps the two native zeta JPEGs distinct from page renders and named licenses', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }) as {
      assets: CuratedAsset[];
    };
    const figures = registry.assets.filter(row => row.asset.includes('/formal-zeta-2026-10-26/'));
    expect(figures).toHaveLength(2);
    for (const row of figures) {
      validateCuratedSourceAsset(row);
      for (const invalid of [{ sha256: '0'.repeat(64) }, { source_asset_sha256: '0'.repeat(64) },
        { source_page: 1 }, { source_pdf_object: 1 }, { derivation: 'pdf-page-render' },
        { license: 'Apache-2.0' }, { rights_status: 'license-verified' },
        { crop_box_px: [0, 0, 1, 1] }, { render_dpi: 300 }])
        expect(() => validateCuratedSourceAsset({ ...row, ...invalid })).toThrow();
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
