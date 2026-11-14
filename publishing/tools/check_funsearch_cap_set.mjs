import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const file = process.argv[2];
if (!file) throw new Error('Usage: node check_funsearch_cap_set.mjs n8_size512.txt');
const payload = readFileSync(file);
assert.equal(createHash('sha256').update(payload).digest('hex'),
  '8d6df45c3039b6a7ee1aae7216b100c3f4c075179dc6471f4f51b968a26d8905');
const points = payload.toString('utf8').trim().split('\n').map(line => JSON.parse(line));
assert.equal(points.length, 512);
const key = point => point.join('');
assert.ok(points.every(point => point.length === 8
  && point.every(x => Number.isInteger(x) && x >= 0 && x < 3)));
const keys = new Set(points.map(key));
assert.equal(keys.size, points.length);
let pairs = 0;
for (let i = 0; i < points.length; i++) {
  for (let j = i + 1; j < points.length; j++) {
    const third = points[i].map((x, dim) => (6 - x - points[j][dim]) % 3);
    assert.ok(!keys.has(key(third)), `collinear triple from pair ${i}, ${j}`);
    pairs++;
  }
}
assert.equal(pairs, 130816);
console.log(`${points.length} distinct 8-dimensional points; all ${pairs} pairs checked`);
