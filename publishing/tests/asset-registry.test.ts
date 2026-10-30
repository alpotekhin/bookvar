import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { extname, resolve } from 'node:path';
import { deflateSync, inflateSync } from 'node:zlib';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';

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
  license_url?: string;
  asset: string;
  derivation?: string;
  provenance_confirmation?: string;
  source_asset?: string;
  source_asset_sha256?: string;
  permission_asset?: string;
  permission_asset_sha256?: string;
  source_stylesheet_asset?: string;
  source_stylesheet_url?: string;
  source_stylesheet_sha256?: string;
  license_evidence?: string;
  license_evidence_sha256?: string;
  source_member?: string;
  source_member_sha256?: string;
  source_cell?: number;
  source_page?: number;
  source_pdf_object?: number;
  source_printed_page?: number;
  render_dpi?: number;
  crop_box_px?: unknown;
  render_page_size_px?: unknown;
  rights_status?: string;
  source_url?: string;
  source_image_url?: string;
  source_code?: string;
  source_code_url?: string;
  source_code_sha256?: string;
  source_page_snapshot?: string;
  source_page_sha256?: string;
  commit?: string;
  metadata_status?: string;
  modifications?: string;
  sha256?: string;
  dimensions_px?: number[];
  frames?: number;
  bytes?: number;
  used_in?: string[];
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

function validateCuratedSourceAsset(entry: CuratedAsset, repositoryRoot = root): void {
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
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
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
      expect(sha256(resolve(repositoryRoot,path!))).toBe(hash);
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
  if (entry.asset.startsWith('00 Учебник/Assets/Figures/curated/preference-2026-10-23/')
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
  // Original PDF recovered and both full-page RGB renders independently reproduced.
  // Provenance is verified; the source slide license remains unverified.
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
    expect(entry.rights_status).toBe('named-lecture-slide-license-unverified');
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
    const sourcePath = 'docs/features/low_precision_training/fp8_delayed_scaling/img/scaling_comparison.svg';
    expect(entry.provenance_confirmation).toBe('pinned-upstream-repository');
    expect(entry.commit).toBe(pin);
    expect(entry.license).toBe('Apache-2.0');
    expect(entry.license_url).toBe(`https://github.com/NVIDIA/TransformerEngine/blob/${pin}/LICENSE`);
    expect(entry.source_url).toBe(`https://github.com/NVIDIA/TransformerEngine/blob/${pin}/${sourcePath}`);
    expect(entry.source_image_url).toBe(`https://raw.githubusercontent.com/NVIDIA/TransformerEngine/${pin}/${sourcePath}`);
    expect(entry.source_asset).toBe('05 Источники/Originals/precision-2026-10-23/scaling_comparison.svg');
    expect(entry.source_stylesheet_asset).toBe('05 Источники/Originals/precision-2026-10-23/diagram-colors.css');
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
    expect(entry.metadata_status).toBe('verified');
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

  // Licensed figures can come from tracked paper/assignment/lecture PDFs,
  // including a pinned paper in an Originals packet, or the
  // original raster cache of an executable lecture.
  if (entry.provenance_confirmation === 'source-license-verified'
    && ['pdf-page-render-crop', 'source-raster-copy'].includes(entry.derivation ?? '')) {
    if (entry.derivation === 'pdf-page-render-crop') {
      expect(entry.source_asset).toMatch(/^05 Источники\/(?:Courses\/[^/]+\/(?:Assignments\/[^/]+|Lectures)|Papers\/Source PDFs|Originals\/[^/]+)\/[^/]+\.pdf$/);
      expect(Number.isInteger(entry.source_page) && Number(entry.source_page) > 0).toBe(true);
      requireNonEmptyString(entry.modifications, 'crop description');
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
    // Legacy raw is ignored and absent from public checkouts; preserve tracked artwork bytes.
    expect(sha256(resolve(root, figure.asset))).toBe('9c5d66c1cfa60f53b7cd36454f2a600b8c5cb82665d16157ba79400090331009');
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
  it('checks all five MLA originals and rejects false rights or changed source metadata',()=>{
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

  it('verifies eleven LSTM/GRU source figures and separates source proof from unconfirmed rights', () => {
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
        if (asset.license_url === null) {
          // Verified source bytes do not imply a verified redistribution license.
          // These three attributed figure quotations explicitly assert none.
          expect(asset.id).toMatch(/^curated-cs336-arc[123]-/);
          expect(asset.rights_status).toMatch(/^limited-educational-quotation/);
          expect(asset.provenance_confirmation).toBe('cs336-source-snapshot; figure-discussed-in-text');
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
    expect(deepseekR1?.used_in).toEqual([
      '00 Учебник/10 Атлас современных архитектур/01 Llama, Qwen и DeepSeek как эволюция блока.md',
      'Papers/DeepSeek-R1 Reasoning via RL.md'
    ]);
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
      const row = registry.assets.find(entry => entry.asset === prefix + name)!;
      expect(row).toBeDefined();
      validateCuratedSourceAsset(row);
      expect(row.source_page).toBe(physical);
      expect(row.source_printed_page).toBe(printed);
      expect(row.render_dpi).toBe(300);
      expect(row.render_page_size_px).toEqual([2481, 3508]);
      expect(row.crop_box_px).toEqual(box);
      expect(row.source_asset_sha256).toBe('3f87b70c64a35d30ec4e565b4d5cfd18fe913002d3b3bed44691c7d0838911bc');
      expect(row.sha256).toBe(hash);
      expect(sha256(resolve(root, row.asset))).toBe(hash);
      expect(pngDimensions(resolve(root, row.asset))).toEqual([box[2]-box[0], box[3]-box[1]]);
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
      expect(entry.rights_status).toBe('named-lecture-slide-license-unverified');
      expect(sha256(resolve(root, entry.asset))).toBe(entry.sha256);
      validateCuratedSourceAsset(entry);
      for (const invalid of [
        {metadata_status: 'needs-confirmation'}, {source_asset: 'missing-source.pdf'},
        {sha256: '0'.repeat(64)}, {rights_status: 'source-license-verified'},
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
  }, 60_000);

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
    const path = resolve(root, '00 Учебник/Assets/Figures/curated/edls-neural-kv-2026-10-26/hf-kv-cache.png');
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
      expect(entry.used_in).toHaveLength(1);
      expect(readFileSync(resolve(root, entry.used_in[0]), 'utf8')).toContain(entry.asset);
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
      expect(row.provenance_confirmation).toBe('user-confirmed-open-materials');
      expect(row.source_asset).toMatch(/^raw\/papers\/external\/vlm-2026\//);
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
    expect(row.provenance_confirmation).toBe('user-confirmed-open-materials');
    expect(row.source_asset).toBe('raw/papers/external/vlm-2026/2503.20215.pdf');
    expect(row.modifications).toContain(archive + 'figure-repairs.json');
    for (const [file, hash] of [[repair.asset, repair.sha256],
      [repair.source_asset, repair.source_asset_sha256], [repair.source_page_snapshot, repair.source_page_sha256],
      [repair.source_raster, repair.source_raster_sha256]])
      expect(sha256(resolve(root, file!))).toBe(hash);
    expect(pngDimensions(resolve(root, repair.asset))).toEqual([1745, 475]);
    expect(pngDimensions(resolve(root, repair.source_raster))).toEqual([1423, 290]);
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


  it('keeps repaired Molmo data and InternVL training figures tied to archived PDF crops', () => {
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
      expect(entry!.provenance_confirmation).toBe('user-confirmed-open-materials');
      expect(entry!.license).toBe('open paper material; reuse confirmed by repository owner');
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
