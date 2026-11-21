import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {loadManifest} from '../adapter/manifest.js';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const audit=JSON.parse(read('docs/audit/2026-10-30-reading-route-review.json'));
const owners=loadManifest(resolve(root,'publishing/navigation.yml')).sections
  .find(x=>x.id==='textbook')!.pages
  .filter(x=>!['textbook/index','textbook/how-to-use'].includes(x.route));
const source=(x:any)=>x.path??x.source;
const rows=[...audit.fullReads,...audit.reusedWholeReads];
const revisions: Record<string,{currentSha256:string;stopBefore:string}> =
  audit.subsequentVerifiedRevisions?.chapters ?? {};
const editorial: typeof revisions = audit.editorialRevisions ?? {};
const sha=(text:string)=>createHash('sha256').update(text).digest('hex');

describe('whole-owner reading receipts',()=>{
  it('pins actual current chapter bytes and excludes non-manifest sources',()=>{
    const canonical=new Set(owners.map(x=>x.source));
    expect(new Set(rows.map(source)).size).toBe(rows.length);
    const seen = new Set<string>();
    const seenEditorial = new Set<string>();
    for(const row of rows){
      const path=source(row),expected=row.sha256??row.currentSha256;
      expect(canonical.has(path),path).toBe(true);
      expect(expected,path).toMatch(/^[a-f0-9]{64}$/);
      const revision=revisions[path];
      const current=read(path), edit=editorial[path];
      let text=current;
      if (edit) {
        seenEditorial.add(path);
        expect(sha(current),path).toBe(edit.currentSha256);
        text=beforeTopicConsolidation(current,path,root,edit.stopBefore);
        expect(sha(text),path).not.toBe(edit.currentSha256);
        expect(() => beforeTopicConsolidation(current+'\nUnreviewed text.\n',
          path,root,edit.stopBefore),path).toThrow();
      }
      if (!revision) {
        expect(sha(text),path).toBe(expected);
        continue;
      }
      seen.add(path);
      expect(sha(text),path).toBe(revision.currentSha256);
      expect(revision.currentSha256,path).not.toBe(expected);
      expect(sha(beforeTopicConsolidation(text,path,root,revision.stopBefore)),path).toBe(expected);
      expect(() => beforeTopicConsolidation(text+'\nUnreviewed text.\n',
        path,root,revision.stopBefore),path).toThrow();
    }
    expect(seen).toEqual(new Set(Object.keys(revisions)));
    expect(seenEditorial).toEqual(new Set(Object.keys(editorial)));
  });
  it('keeps the progress counters equal to the hash-pinned read set',()=>{
    expect(audit.currentLocalVerification.readReceipt).toEqual({
      currentHashMatches:rows.length,
      outsideTextbookManifest:0,
      unreadOwnersInRoutePass:owners.length-rows.length,
    });
    expect(audit.reviewLimits).toContain(audit.fullReads.length+' complete canonical-owner reads');
  });
});

