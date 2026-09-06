import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
const courseRoot = resolve(root, '05 Источники/Courses/Stanford CS336 Spring 2026');
const revision = '8b59b50730766695c2ffedd1a79c50cd09b9eb91';
const multimodalRoute = 'textbook/multimodal/unified-sequence-chameleon';

const multimodalAssets = [
  'siglip-parallelism.png',
  'llava-gen.png',
  'llava-onevision.png',
  'llava-onevision-anyres.png',
  'llava-onevision-modalities.png',
  'llava-onevision-training.png',
  'llava-onevision-transfer-s1.png',
  'llava-onevision-transfer-s2.png',
  'qwen-vl-stages.png',
  'qwen2-vl-architecture.png',
  'qwen3-vl.png',
  'qwen3-vl-pretraining.png',
  'chameleon.png',
  'chameleon-example.png',
  'vq-vae.png'
];

const replacements = [
  'vit-figure1-patch-sequence.png',
  'clip-figure1-training-and-zero-shot.png',
  'llava-figure1-projector.png',
  'qwen2vl-figure3-mrope.png'
];

function sha256(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

describe('Stanford CS336 Lecture 17 publication integration', () => {
  it('closes all ten source units and seven visual decisions', () => {
    const coverage = JSON.parse(readFileSync(resolve(courseRoot, 'coverage.yml'), 'utf8')) as {
      rows: Array<{
        source_object: string;
        source_unit: string;
        disposition: string;
        destination?: string;
        destination_anchor?: string;
      }>;
    };
    const visuals = JSON.parse(readFileSync(resolve(courseRoot, 'visuals.yml'), 'utf8')) as {
      rows: Array<{ source_object: string; disposition: string; destination_anchor?: string }>;
    };
    const lectureCoverage = coverage.rows.filter(({ source_object }) => source_object === 'lecture-17');
    const lectureVisuals = visuals.rows.filter(({ source_object }) => source_object === 'lecture-17');

    expect(lectureCoverage).toHaveLength(10);
    expect(lectureCoverage.filter(({ disposition }) => disposition === 'integrated')).toHaveLength(9);
    expect(lectureCoverage.filter(({ disposition }) => disposition === 'excluded')).toHaveLength(1);
    expect(lectureCoverage.some(({ disposition }) => disposition === 'source-only')).toBe(false);
    expect(lectureVisuals).toHaveLength(7);
    expect(lectureVisuals.filter(({ disposition }) => disposition === 'integrated')).toHaveLength(6);
    expect(lectureVisuals.filter(({ disposition }) => disposition === 'excluded')).toHaveLength(1);
    expect(lectureVisuals.some(({ disposition }) => disposition === 'source-only')).toBe(false);
    expect(Object.fromEntries(lectureCoverage.map((row) => [row.source_unit, {
      disposition: row.disposition,
      destination: row.destination,
      destination_anchor: row.destination_anchor
    }]))).toEqual({
      'lecture-17-section-2-lecture-17-multimodal-models': {
        disposition: 'integrated',
        destination: '00 Учебник/16 Multimodal Models/64 Мультимодальные модели.md',
        destination_anchor: 'две-задачи-понять-чужую-модальность-и-породить-её'
      },
      'lecture-17-figure-step-6-rendering-1': {
        disposition: 'excluded',
        destination: undefined,
        destination_anchor: undefined
      },
      'lecture-17-contrastive-vision-language-training': {
        disposition: 'integrated',
        destination: '00 Учебник/16 Multimodal Models/64 Мультимодальные модели.md',
        destination_anchor: 'clip-и-siglip-две-формы-контрастивного-обучения'
      },
      'lecture-17-clip-siglip-training-sequence': {
        disposition: 'integrated',
        destination: '00 Учебник/16 Multimodal Models/64 Мультимодальные модели.md',
        destination_anchor: 'clip-и-siglip-две-формы-контрастивного-обучения'
      },
      'lecture-17-llava-onevision-progression': {
        disposition: 'integrated',
        destination: '00 Учебник/16 Multimodal Models/64a Connectors и fusion.md',
        destination_anchor: 'от-llava-к-llava-onevision-что-изменилось-кроме-projector'
      },
      'lecture-17-multimodal-tokenization-failure-modes': {
        disposition: 'integrated',
        destination: '00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции.md',
        destination_anchor: 'где-теряется-информация-crop-tiling-merge-и-фиксированный-budget'
      },
      'lecture-17-qwen-vl-training-stages': {
        disposition: 'integrated',
        destination: '02 Атлас моделей/Семейства/Мультимодальные семейства.md',
        destination_anchor: 'qwen-vl-lineage'
      },
      'lecture-17-figure-step-210-rendering-1': {
        disposition: 'integrated',
        destination: '00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon.md',
        destination_anchor: '4-одна-авторегрессионная-последовательность'
      },
      'lecture-17-figure-step-211-rendering-1': {
        disposition: 'integrated',
        destination: '00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon.md',
        destination_anchor: '8-что-даёт-единый-словарь-а-чего-он-не-гарантирует'
      },
      'lecture-17-figure-step-216-rendering-1': {
        disposition: 'integrated',
        destination: '00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon.md',
        destination_anchor: '3-как-изображение-становится-дискретными-кодами'
      }
    });
  });

  it('publishes 64h once and preserves the 64f to 65 linear order', () => {
    const navigation = parse(readFileSync(resolve(root, 'publishing/navigation.yml'), 'utf8')) as {
      sections: Array<{
        id: string;
        pages: Array<{ route: string }>;
        sidebar?: Array<{ label: string; items?: Array<{ route: string }> }>;
      }>;
    };
    const textbook = navigation.sections.find(({ id }) => id === 'textbook')!;
    const pageRoutes = textbook.pages.map(({ route }) => route);
    const module = textbook.sidebar?.find(({ label }) => label === 'XIII. Мультимодальные модели');
    const sidebarRoutes = module?.items?.map(({ route }) => route) ?? [];
    const expected = [
      'textbook/multimodal/evaluation-serving',
      'textbook/multimodal/diffusion-flow-images',
      multimodalRoute,
      'textbook/agents/tool-use'
    ];

    expect(pageRoutes.filter((route) => route === multimodalRoute)).toHaveLength(1);
    expect(sidebarRoutes.filter((route) => route === multimodalRoute)).toHaveLength(1);
    expect(pageRoutes.slice(pageRoutes.indexOf(expected[0]), pageRoutes.indexOf(expected[0]) + 4))
      .toEqual(expected);
    expect(sidebarRoutes.slice(-3)).toEqual(expected.slice(0, 3));

    const moduleMap = readFileSync(
      resolve(root, '00 Учебник/16 Multimodal Models/01 Vision-language и omni models.md'),
      'utf8'
    );
    const textbookIndex = readFileSync(resolve(root, '00 Учебник/_index.md'), 'utf8');
    expect(moduleMap.match(/64h Единая мультимодальная последовательность и Chameleon/g)).toHaveLength(1);
    expect(textbookIndex.match(/64h Единая мультимодальная последовательность и Chameleon/g)).toHaveLength(1);
  });

  it('registers all fifteen new assets and four high-resolution replacements at the pinned revision', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), {
      merge: true
    }) as { assets: Array<Record<string, unknown>> };
    const byPath = new Map(registry.assets.map((entry) => [entry.asset, entry]));
    const paths = [
      ...multimodalAssets.map((name) =>
        `00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/${name}`
      ),
      ...replacements.map((name) => `00 Учебник/Assets/Figures/curated/vlm-2026/${name}`)
    ];

    for (const path of paths) {
      const entry = byPath.get(path);
      expect(entry, path).toBeDefined();
      expect(entry?.commit, path).toBe(revision);
      expect(entry?.source_url, path).toMatch(
        new RegExp(`^https://github\\.com/stanford-cs336/lectures/blob/${revision}/`)
      );
      expect(entry?.sha256, path).toBe(sha256(resolve(root, path)));
      expect(entry?.provenance_confirmation, path).toBe('pinned-upstream-repository');
    }
  });
});
