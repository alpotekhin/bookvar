import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import matter from 'gray-matter';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const text=(p:string)=>read(p).toString(),sha=(s:Buffer|string)=>createHash('sha256').update(s).digest('hex');
const own='2026-11-03-word-embeddings-data-additions.json',stop='2026-11-03-neural-lm-bengio-additions.json';
const journal=JSON.parse(text('docs/audit/'+own)),row=journal.files[0];
const current=beforeTopicConsolidation(text(row.source),row.source,root,own);
const prior=beforeTopicConsolidation(text(row.source),row.source,root,stop);
const programs=(s:string)=>[...s.matchAll(/^\x60{3}python\n([\s\S]*?)^\x60{3}\s*$/gm)].map(m=>m[1]);

describe('word-vector course data mechanisms and canonical navigation',()=>{
  it('restores the exact complete preceding owner and both indexes',()=>{
    expect(journal.files).toHaveLength(3);
    for(const entry of journal.files){
      expect(sha(beforeTopicConsolidation(text(entry.source),entry.source,root,own))).toBe(entry.after_sha256);
      expect(sha(beforeTopicConsolidation(text(entry.source),entry.source,root,stop))).toBe(entry.before_sha256);
      expect(()=>beforeTopicConsolidation(text(entry.source)+'UNREVIEWED',entry.source,root,stop)).toThrow();
    }
    expect(sha(prior)).toBe('53dba94646abb1903ae0ca729e6caea54b99b9c6a5a4a9fb63ede9dda7df1bab');
  });
  it('retains all five prior complete programs, equations and eight source images',()=>{
    expect(programs(current)).toHaveLength(7);
    expect(programs(current).filter(x=>!x.includes('def get_centers_and_contexts')&&!x.includes('def batchify')))
      .toEqual(programs(prior));
    const math=(s:string)=>[...s.matchAll(/^\$\$\n([\s\S]*?)^\$\$/gm)].map(m=>m[1]);
    expect(math(current).filter(x=>!x.includes('pair included')&&!x.includes('q_C(n)'))).toEqual(math(prior));
    const images=(s:string)=>[...s.matchAll(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(m=>m[1]);
    expect(images(current)).toEqual(images(prior));
    expect(images(current)).toHaveLength(8);
    const assets=parse(text('05 Источники/asset-registry.yml')).assets;
    for(const prefixed of images(current)){
      const path=prefixed.replace(/^02 Areas\/ML & DL\//,'');
      const entries=assets.filter((a:any)=>a.asset===path);
      expect(entries,path).toHaveLength(1);
      expect(sha(read(path)),path).toBe(entries[0].sha256);
    }
    expect(current).toContain('<a id="краткие-итоги"></a>');
  });
  it('compares the original and adapted source functions with the archived source', { tags: ['source-archive'] }, ()=>{
    const source=journal.sourceComparison[1];
    expect(sha(read(source.path))).toBe(source.sha256);
    const raw=text(source.path);
    const original=raw.match(/def get_centers_and_contexts\(corpus, max_window_size\):[\s\S]*?    return centers, contexts/)![0];
    expect(programs(current).find(x=>x.includes('def get_centers_and_contexts'))).toContain(original);
    const batch=raw.match(/def batchify\(data\):[\s\S]*?contexts_negatives\), d2l\.tensor\(masks\), d2l\.tensor\(labels\)\)/)![0];
    const adapted=batch.replace('d2l.reshape(d2l.tensor(centers), (-1, 1))',
      'torch.tensor(centers).reshape((-1, 1))').replaceAll('d2l.tensor(','torch.tensor(');
    expect(programs(current).find(x=>x.includes('def batchify'))).toContain(adapted);
  });
  it('pins the existing public license and the worked-example details',()=>{
    const source=journal.sourceComparison[1];
    expect(text(source.licensePath)).toContain('Creative Commons Attribution-ShareAlike 4.0 International Public License');
    for(const phrase of ['0.8','0.2','context-dependent rejection','Repeated context positions',
      'zero label alone','global mean is $16/11$','means is $1.5$'])
      expect(current).toContain(phrase);
  });
  it('indexes twenty-five mechanisms in both locales without a translation pass',()=>{
    const meta=matter(current).data;
    expect(meta.source_language).toBe('mixed');
    expect(meta.reference_terms).toHaveLength(25);
    for(const term of meta.reference_terms){
      expect(current.split('<a id="'+term.anchor+'"></a>')).toHaveLength(2);
      for(const index of ['01 Справочник/_index.md','en/01 Reference/_index.md'])
        expect(text(index)).toContain('[['+row.source.slice(0,-3)+'#'+term.anchor+'|');
    }
    expect(current).not.toContain('Пять исполняемых блоков выше');
  });
});
