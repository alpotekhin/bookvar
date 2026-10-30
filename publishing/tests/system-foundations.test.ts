import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
const file = '00 Учебник/10 ML Systems/01 Модель как часть системы.md';
const body = readFileSync(resolve(root, file), 'utf8');
const digest = (path: string) => createHash('sha256')
  .update(readFileSync(resolve(root, path))).digest('hex');

describe('model as part of a system', () => {
  it('checks the actual stage table, burst table and periodic schedule', () => {
    const stages = [...body.matchAll(/\| (?:Получение и подготовка изображения на CPU|Передача входного тензора на GPU|Выполнение модели|Выбор решения и передача команды) \| (\d+) мс \|/g)]
      .map(row => Number(row[1]));
    expect(stages).toEqual([8, 4, 20, 3]);
    const total = stages.reduce((sum, time) => sum + time, 0);
    expect(total).toBe(35);
    expect(total / (total - stages[2] / 2)).toBe(1.4);
    expect(total / (total - stages[2])).toBeCloseTo(2.333333333333333);

    const rows = [...body.matchAll(/\| (\d+) мс \| (\d+) мс \| (\d+) мс \|/g)]
      .map(row => row.slice(1).map(Number));
    expect(rows).toHaveLength(3);
    let finish = 0;
    for (const [arrival, publishedFinish, latency] of rows) {
      finish = Math.max(finish, arrival) + total;
      expect(publishedFinish).toBe(finish);
      expect(latency).toBe(finish - arrival);
    }

    // Independent dedicated resources, FIFO, unlimited waiting buffers.
    // Each cell waits for both this frame's input and this stage's resource.
    const ends = stages.map(() => 0);
    const completions: number[] = [];
    for (let group = 0; group < 8; group++) {
      for (const offset of [0, 1, 2]) {
        let ready = group * 120 + offset;
        stages.forEach((service, j) => {
          ends[j] = Math.max(ready, ends[j]) + service;
          ready = ends[j];
        });
        completions.push(ready - group * 120);
      }
    }
    expect(completions).toEqual(Array.from({ length: 8 }, () => [35, 55, 75]).flat());
    expect(completions.slice(0, 3).map((end, i) => end - i)).toEqual([35, 54, 73]);
    expect(completions.filter((end, i) => end - i % 3 > 50)).toHaveLength(16);
    expect(4e6 / 1e9 * 1000).toBe(4);
    expect(2e9 / (1e11 * .5) * 1000).toBe(40);
    expect(20 * .047).toBeCloseTo(.94);
  });

  it('keeps all six original images and matches the recorded publication bytes', () => {
    const images = [...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(row => row[1]);
    expect(images).toHaveLength(6);
    const checks = {
      '00 Учебник/Assets/Figures/curated/systems-foundations-review-2026-09/ml-lifecycle.png': '05fdb7e3a7c16485cc708f34393073ad6079f5d933aefbb9b129f9edb33adf37',
      '00 Учебник/Assets/Figures/curated/systems-visuals-2026-09/mit-amdahl.png': '4d9db2b5cb2a6ccd2eca361b79f5a007f7029282169288bc31596668cc456f38',
      '00 Учебник/Assets/Figures/curated/systems-visuals-2026-09/mit-pipeline.png': '4df0829852360d1a96d5186e866740ee1232ef2e1effc040ca115f425c748301',
      'Assets/Sources/Harvard ML Systems/vol1/model_serving/2f10e92057738ef1e4a50e524fddf5389d61d795.svg': '4a5bd2caaa8c19de2f9d7e1a2efaa52829595fbf5cd88fd315b74f8f99b56888',
      'Assets/Sources/Harvard ML Systems/vol1/ml_systems/c491ee41569a92f002bf915150731598e0d64e26.svg': '51eab06be405c5f84400505ad57a3255ac913b4c9d36f4e9484cde4b98a1b888',
      'Assets/Sources/Harvard ML Systems/vol1/ml_workflow/3b022a3ffa6f028fe2c2c3165c292a16a3a9e0ff.svg': 'c4d3aadc33da2a2a1146ac8276206ca31218c4d083ce948810bd92b43490de18'
    };
    expect([...images].sort()).toEqual(Object.keys(checks).sort());
    for (const [path, hash] of Object.entries(checks)) expect(digest(path)).toBe(hash);
  });

  it('retains source attribution, scope and adjacent chapters', () => {
    expect(body).toContain('45ecc8d82fcae70c149cdce550d3b3d3411df913');
    expect(body).toContain('CC BY-NC-SA 4.0');
    expect(body).toContain('не результат измерения конкретной камеры или GPU');
    expect(body).toContain('Если независимые части передачи и вычисления');
    expect(body).toContain('_index#Углублённые модули|');
    expect(body).toContain('02 GPU, CUDA и иерархия памяти|');
  });
});
