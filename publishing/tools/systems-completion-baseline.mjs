import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
const defaultRoot=resolve(import.meta.dirname,'../..');
const sha=s=>createHash('sha256').update(s).digest('hex');

export function beforeSystemsCompletion(text,source,root=defaultRoot){
  const journal=JSON.parse(readFileSync(resolve(root,
    'docs/audit/2026-10-29-systems-completion-additions.json'),'utf8'));
  const row=journal.files.find(r=>r.source===source);
  if(!row || sha(text)===row.before_sha256) return text;
  for(const block of row.blocks){
    const re=new RegExp(`<!-- systems-completion: ${block.id}:start -->[\\s\\S]*?<!-- systems-completion: ${block.id}:end -->\\n\\n`,'g');
    const found=[...text.matchAll(re)];
    if(found.length!==1 || sha(found[0][0])!==block.sha256)
      throw Error(`Unreviewed systems completion block: ${source}: ${block.id}`);
    text=text.replace(found[0][0],'');
  }
  for(const c of row.frontmatter_changes){
    if(!text.includes(c.new))throw Error(`Unrecognized systems frontmatter: ${source}`);
    text=text.replace(c.new,c.old);
  }
  return text;
}
