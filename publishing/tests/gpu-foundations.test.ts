import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import YAML from 'yaml';
import { describe, expect, it } from 'vitest';
import { beforeSystemsCompletion } from '../tools/systems-completion-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const file = '00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md';
// The additive systolic explanation has its own exact-revision checks.
const body = beforeSystemsCompletion(readFileSync(resolve(root, file), 'utf8'), file, root);

describe('GPU foundations explanation', () => {
  it('spells out the actual XOR bank map and its diagonal limitation', () => {
    const section = body.split('### Swizzling:')[1].split('## Блочное')[0];
    const rows = [...section.matchAll(/\| ([0-3]) \| ([0-3]) \| ([0-3]) \| ([0-3]) \| ([0-3]) \|/g)]
      .map(row => row.slice(1).map(Number));
    expect(rows).toHaveLength(4);
    rows.forEach(([r, ...banks]) => expect(banks).toEqual([0, 1, 2, 3].map(c => c ^ r)));
    expect(section).toContain('и при записи, и при чтении');
    expect(section).toContain('На диагонали $r=c$ все элементы снова попадают в банк 0');
  });

  it('keeps an executable boundary example and separates it from GPU claims', () => {
    expect(body.match(/~~~python\n/g)).toHaveLength(2);
    expect(body).toContain('def tiled_matmul(A, B, tile=2):');
    expect(body).toContain('[[58., 64.], [139., 154.]]');
    expect(body.replace(/\s+/g, ' ').includes('время выполнения ничего не говорит о скорости CUDA-ядра')).toBe(true);
    expect(body).toContain('`sm_100a`/`sm_100f`');
    expect(body).toContain('128\\cdot32\\cdot4=16384');
    expect(body).toContain('памяти, не номера потоков warp');
    expect(body).toContain('не является полным исполняемым приложением или результатом GPU-бенчмарка');
  });

  it('explains device placement without claiming that the CPU fallback exercised CUDA', () => {
    for (const detail of ['id="torch-device-placement"', 'source_cpu.to(target_device)',
      'with torch.device(target_device)', 'same_cpu is source_cpu',
      'source_cpu.to("cpu", copy=True)', 'CUDA transfer executed:', 'CPU-ветка',
      'явный аргумент `device="cpu"`', 'Уже существующие тензоры этот контекст не переносит'])
      expect(body).toContain(detail);
    const fm = YAML.parse(body.split('---')[1]);
    expect(fm.source_unit_id).toContain('lecture-02-tensor-device-placement');
  });

  it('retains all fifteen source figures, source anchors and the chapter sequence', () => {
    const images = [...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(row => row[1]);
    expect(images).toHaveLength(15);
    const registry = YAML.parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }).assets;
    for (const image of images) {
      expect(existsSync(resolve(root, image))).toBe(true);
      expect(registry.find((row: { asset: string }) => row.asset === image).used_in).toContain(file);
    }
    expect(body.match(/id="cs336-systems-gpu"/g)).toHaveLength(1);
    expect(body).toContain('01 Модель как часть системы|Модель как часть системы');
    expect(body).toContain('03 Измерение производительности и roofline|Измерение производительности и roofline');
  });
});
