import { beforeReaderRouteNavigation } from './reader-route-baseline.mjs';
import {readFileSync,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const root=resolve(import.meta.dirname,'../..');
const hash=data=>createHash('sha256').update(data).digest('hex');
export function checkSectionMap(map,base=root) {
  const errors=[];
  const read=path=>readFileSync(resolve(base,path));
  if(hash(read(map.source.path))!==map.source.sha256) errors.push('Source revision changed: re-review required');
  const sourceLines=read(map.source.path).toString().split('\n').length-1;
  if(sourceLines!==map.source.lines) errors.push('Source line count changed');
  let next=1;
  const ids=new Set();
  for(const row of map.rows) {
    if(row.start!==next||row.end<row.start) errors.push('Gap/overlap at '+row.id);
    next=row.end+1;
    if(ids.has(row.id)) errors.push('Duplicate row '+row.id);
    ids.add(row.id);
    if(!row.explanation||!row.evidence.length) errors.push('Missing editorial decision/evidence: '+row.id);
    for(const target of row.targets) if(!map.targets[target]) errors.push('Unknown destination '+target);
    for(const evidence of row.evidence) if(!existsSync(resolve(base,evidence))) errors.push('Missing evidence '+evidence);
  }
  if(next!==map.source.lines+1) errors.push('Source end is uncovered');
  for(const [id,target] of Object.entries(map.targets)) {
    const reviewed = beforeReaderRouteNavigation(read(target.path).toString(), target.path, base);
    if(hash(reviewed)!==target.sha256) errors.push('Stale reviewed destination: '+id);
    if(reviewed.split('\n').length-1!==target.lines) errors.push('Destination line count: '+id);
    if(!read(target.review_evidence).toString().includes(target.sha256)) errors.push('Review does not identify revision: '+id);
  }
  const figures=JSON.parse(read(map.figures));
  if(figures.figures.length!==24) errors.push('Native-figure decisions changed');
  return {scope:'Integrity of explicit editorial decisions, not an automated semantic-completeness test',
    rows:map.rows.length,destinations:Object.keys(map.targets).length,source_lines:sourceLines,errors};
}

if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href) {
  const map=JSON.parse(readFileSync(resolve(root,'docs/audit/2026-10-27-nn-section-map.json')));
  const result=checkSectionMap(map);
  console.log(JSON.stringify(result,null,2));
  if(result.errors.length) process.exitCode=1;
}
