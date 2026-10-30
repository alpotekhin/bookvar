import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,it,expect} from 'vitest';
import {beforeHardwareSourceCompletion} from '../tools/hardware-source-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path),'utf8');
const sha=(text:string)=>createHash('sha256').update(text).digest('hex');
const journal=JSON.parse(read('docs/audit/2026-10-29-hardware-source-comparison.json'));
const original=read(journal.source.path);
const chapter=read(journal.addition.source);
const slug=(text:string)=>text.toLowerCase().replace(/\s+/g,'-').replace(/[^\p{L}\p{N}_-]/gu,'');

describe('Hardware source full comparison',()=>{
  it('pins the complete source and maps all 113 headings without range gaps',()=>{
    expect(sha(original)).toBe(journal.source.sha256);
    const lines=original.trimEnd().split('\n');
    expect(lines).toHaveLength(5935);
    const headings=lines.flatMap((line,index)=>{
      const match=line.match(/^#{1,6} (.+?) \{#([^}]+)\}$/);
      return match?[{line:index+1,title:match[1],anchor:match[2]}]:[];
    });
    expect(headings).toHaveLength(113);
    expect(journal.sections).toHaveLength(headings.length);
    for(const [i,h]of headings.entries()){
      const row=journal.sections[i];
      expect(row.source_heading).toBe(h.title);
      expect(row.source_anchor).toBe(h.anchor);
      expect(row.source_range).toEqual([h.line,(headings[i+1]?.line??5936)-1]);
      expect(['covered-existing','added','inapplicable']).toContain(row.status);
      expect(row.comparison.length).toBeGreaterThan(40);
      expect(row.destination_ids.length).toBeGreaterThan(0);
      for(const id of row.destination_ids)expect(journal.destinations[id]).toBeDefined();
    }
    expect(journal.source.full_read.ranges).toEqual([[1,5935]]);
    expect(journal.source.full_read.reader_chunks_read).toEqual(Array.from({length:18},(_,i)=>i));
    expect(journal.source.full_read.excluded_hidden_blocks).toHaveLength(56);
  });
  it('links to real canonical headings and registered routes',()=>{
    const nav=read('publishing/navigation.yml');
    const routes=new Map([...nav.matchAll(/source: (.+)\n(?:\s+source_en: .+\n)?\s+route: (.+)/g)].map(m=>[m[1],m[2]]));
    for(const d of Object.values(journal.destinations) as any[]){
      expect(routes.get(d.path),d.path).toBe(d.route);
      const headings=[...read(d.path).matchAll(/^#{1,6} (.+)$/gm)].map(m=>m[1]);
      expect(headings,d.path).toContain(d.heading);
      expect(d.anchor).toBe(slug(d.heading));
      expect(d.read_ranges.length).toBeGreaterThan(0);
    }
  });
  it('retains exact licensed source paragraphs apart from declared publication markup',()=>{
    const lines=original.split('\n');
    const normalize=(s:string)=>s.replace(/\\index\{[^}]*\}/g,'').replace(/\[\^fn-[^\]]+\]/g,'').replace(/ \[@[^\]]+\]/g,'');
    for(const row of journal.source_excerpts){
      const pattern=new RegExp('<!-- hardware-excerpt: '+row.id+':start -->\\n> ([\\s\\S]*?)\\n<!-- hardware-excerpt: '+row.id+':end -->');
      expect(chapter.match(pattern)?.[1],row.id).toBe(normalize(lines[row.source_line-1]));
    }
    expect(journal.source_excerpts).toHaveLength(4);
    expect(chapter).toContain('not a GPU, TPU or mobile-device benchmark');
  });
  it('preserves the prior parent revision byte-for-byte and rejects unreviewed edits',()=>{
    const row=journal.addition;
    expect(sha(chapter)).toBe(row.after_sha256);
    const before=beforeHardwareSourceCompletion(chapter,row.source,root);
    expect(sha(before)).toBe(row.before_sha256);
    expect(beforeHardwareSourceCompletion(before,row.source,root)).toBe(before);
    expect(()=>beforeHardwareSourceCompletion(chapter.replace('Matching an operation','Matching any operation'),row.source,root)).toThrow();
    expect(sha(beforeHardwareSourceCompletion(chapter+'unreviewed outside edit',row.source,root))).not.toBe(row.before_sha256);
    for(const id of ['hardware-im2col','hardware-layout'])expect(chapter).toContain('<!-- checked-example: '+id+' -->');
  });
  it('records actual figure treatment and source caveats without claiming hardware validation',()=>{
    expect(journal.figures).toHaveLength(14);
    expect(journal.source_issues.length).toBeGreaterThan(35);
    for(const f of journal.figures){
      expect(original).toContain(f.source_id);
      expect(f.reason.length).toBeGreaterThan(40);
    }
    expect(journal.verification.scope).toContain('No GPU/TPU/NPU/migration/energy benchmark');
  });
});
