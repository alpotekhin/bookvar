import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,it,expect} from 'vitest';
import {beforeSystemsCompletion} from '../tools/systems-completion-baseline.mjs';
import {beforeHardwareSourceCompletion} from '../tools/hardware-source-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(s:string)=>createHash('sha256').update(s).digest('hex');
const rows=JSON.parse(read('docs/audit/2026-10-29-systems-completion-additions.json')).files;
describe('Systolic execution and compression additions',()=>{
  it('preserves the accepted chapters outside separately checked additions',()=>{
    for(const row of rows){
      const text=beforeHardwareSourceCompletion(read(row.source),row.source,root);
      expect(sha(text)).toBe(row.after_sha256);
      expect(sha(beforeSystemsCompletion(text,row.source,root))).toBe(row.before_sha256);
      expect(()=>beforeSystemsCompletion(text.replace('<!-- systems-completion:',
        '<!-- invalid-completion:'),row.source,root)).toThrow();
      expect(sha(beforeSystemsCompletion(text+'Unreviewed outside edit',row.source,root)))
        .not.toBe(row.before_sha256);
    }
  });
  it('keeps the working assumptions and the source illustrations',()=>{
    const gpu=beforeHardwareSourceCompletion(read(rows[0].source),rows[0].source,root), compression=read(rows[1].source);
    expect(gpu.match(/<!-- checked-example: hardware-/g)).toHaveLength(2);
    expect(compression.match(/<!-- checked-example:/g)).toHaveLength(17);
    expect(compression.match(/!\[\[/g)).toHaveLength(12);
    for(const phrase of ['M+K+N-2', 'not measured HBM transactions',
      'not an emulator', 'tpu-load-weights.gif', 'tpu-dataflow.gif'])
      expect(gpu).toContain(phrase);
    for(const phrase of ['original initialization', 'original model on **both** subsets',
      'not the deployment result', 'KV-cache', 'pruning-artifact', 'nas-evolution'])
      expect(compression).toContain(phrase);
  });
});
