import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeSelectionGapsAddition} from '../tools/selection-gaps-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p),'utf8');
const sha=(value:string|Buffer)=>createHash('sha256').update(value).digest('hex');
const journal=JSON.parse(read('docs/audit/2026-10-29-selection-gaps-additions.json'));
describe('Image deduplication and cold-ID explanation additions',()=>{
  it('preserves every prior chapter byte and rejects unreviewed changes',()=>{
    for(const row of journal.files){
      const current=read(row.source);
      expect(sha(current)).toBe(row.after_sha256);
      expect(sha(beforeSelectionGapsAddition(current,row.source))).toBe(row.before_sha256);
      expect(()=>beforeSelectionGapsAddition(current.replace(`<!-- ${row.marker}-addition: start -->`,
        `<!-- ${row.marker}-addition: start -->\nUnreviewed change.`),row.source)).toThrow();
      expect(sha(beforeSelectionGapsAddition(current+'\nUnreviewed outside change.\n',row.source)))
        .not.toBe(row.before_sha256);
    }
  });
  it('keeps both entire source images, licenses and provenance byte-exact',()=>{
    const assets=JSON.parse(read('00 Учебник/Assets/Figures/curated/image-id-selection-2026-10-29/assets.json'));
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(assets).toHaveLength(2);
    for(const row of assets){
      expect(registry.find((a:any)=>a.id===row.id)).toEqual(row);
      for(const [path,hash] of [[row.asset,row.sha256],[row.source_asset,row.source_asset_sha256],
        [row.source_page_snapshot,row.source_page_sha256],[row.license_evidence,row.license_evidence_sha256]])
        expect(sha(readFileSync(resolve(root,path)))).toBe(hash);
      const data=readFileSync(resolve(root,row.asset));
      expect(data).toEqual(readFileSync(resolve(root,row.source_asset)));
      expect([data.readUInt32BE(16),data.readUInt32BE(20)]).toEqual(row.dimensions_px);
      expect(readFileSync(resolve(root,row.public_license_copy)))
        .toEqual(readFileSync(resolve(root,row.license_evidence)));
      expect(row.crop_box_px).toBeUndefined();
      expect(read(row.used_in[0])).toContain(row.asset);
    }
  });
  it('retains explicit decision boundaries and unchanged original-language excerpts',()=>{
    const image=read(journal.files[0].source),ids=read(journal.files[1].source);
    for(const text of ['pixel-level matching','red–blue','any earlier record',
      'Equality at the threshold','caption diversity','candidate-recall']) expect(image).toContain(text);
    for(const text of ['zero item IDs','OOV','optimizer references','step counters',
      'known-but-pruned']) expect(ids).toContain(text);
    expect(ids).toContain('construction cell instantiates `MCHManagedCollisionModule`');
    for(const [chapter,source] of [[image,'imagehash-README.rst'],[ids,'torchrec-zch-README.md']]){
      const addition=chapter.match(/<!-- (?:image-dedup|cold-ids)-addition: start -->[\s\S]*?<!-- (?:image-dedup|cold-ids)-addition: end -->/)![0];
      const excerpt=addition.match(/(?:^> .*\n)+/m)![0].split('\n').map(x=>x.replace(/^> /,'')).join(' ');
      const normalize=(s:string)=>s.replace(/\s+/g,' ').trim();
      expect(normalize(read('05 Источники/Originals/image-id-selection-2026-10-29/'+source))).toContain(normalize(excerpt));
    }
  });
});
