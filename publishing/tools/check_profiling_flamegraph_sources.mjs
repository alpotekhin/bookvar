import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'../..');
const m=JSON.parse(readFileSync(resolve(root,'publishing/tools/profiling-flamegraph-sources.json'),'utf8'));
const hash=b=>createHash('sha256').update(b).digest('hex');
const originals=[];
for(const row of m.sources){
  const local=readFileSync(resolve(root,row.path));
  if(hash(local)!==row.sha256)throw Error('Source hash mismatch: '+row.path);
  if(process.argv.includes('--upstream')){
    const response=await fetch(row.url);
    if(!response.ok)throw Error('HTTP '+response.status+': '+row.url);
    if(!local.equals(Buffer.from(await response.arrayBuffer())))throw Error('Upstream differs: '+row.path);
  }
  originals.push(local.toString('utf8'));
}
const svg=originals[0],figure=readFileSync(resolve(root,m.figure));
if(figure.toString('utf8')!==svg||hash(figure)!==m.figure_sha256)throw Error('Source SVG changed');
if(!svg.includes('viewBox="0 0 1200 182"')||!svg.includes('var inverted = true;'))
  throw Error('Source geometry or orientation changed');
if(/\b(?:fetch|XMLHttpRequest|WebSocket|sendBeacon|localStorage|sessionStorage)\b|document\.cookie/.test(svg))
  throw Error('Unexpected I/O in the original graph');
const frames=[...svg.matchAll(/<g><title>(.*?)<\/title><rect x="([^"]+)" y="([^"]+)" width="([^"]+)" height="([^"]+)"/g)]
  .map(x=>({title:x[1],x:+x[2],y:+x[3],width:+x[4],count:+x[1].match(/([\d,]+) samples/)[1].replaceAll(',','')}));
const frame=text=>{const found=frames.filter(x=>x.title.startsWith(text));if(found.length!==1)throw Error('Ambiguous '+text);return found[0]};
const all=frame('all ('),caller79=frame('calculate_similar_artists (lastfm.py:79)'),
  caller95=frame('calculate_similar_artists (lastfm.py:95)'),
  score209=frame('_get_similarity_score (implicit/recommender_base.py:209)'),
  argpartition=frame('argpartition ('),wrapfunc=frame('_wrapfunc (');
for(const [key,value]of Object.entries({all,caller79,caller95,score209,argpartition,wrapfunc}))
  if(value.count!==m.sample_counts[key])throw Error('Source count differs: '+key);
for(const [parent,child]of [[all,caller79],[all,caller95],[score209,argpartition],[argpartition,wrapfunc]])
  if(child.y<=parent.y||child.x<parent.x||child.x+child.width>parent.x+parent.width||child.count>parent.count)
    throw Error('Invalid source ancestry');
const percentages=[caller79,caller95,score209,argpartition,wrapfunc].map(x=>(100*x.count/all.count).toFixed(2));
if(JSON.stringify(percentages)!==JSON.stringify(['51.50','47.39','28.14','28.10','28.10']))
  throw Error('Incorrect percentages');
const parentSamples=new Set(Array.from({length:814},(_,i)=>i));
const childSamples=new Set(Array.from({length:813},(_,i)=>i));
const union=new Set([...parentSamples,...childSamples,...childSamples]);
if(union.size!==814||parentSamples.size+2*childSamples.size!==2440)throw Error('Double-count example');
const first='DDDDDDWW',second='DDD WDDD W'.replaceAll(' ','');
for(const sequence of [first,second])if(sequence.length!==8||[...sequence].filter(x=>x==='D').length!==6)
  throw Error('Order counterexample changed');
if(first===second)throw Error('Order example is not distinct');
const chapter=readFileSync(resolve(root,m.chapter),'utf8');
for(const fact of ['2893','1490','1371','814','813','51.50%','47.39%','28.14%','28.10%',
 '[D,D,D,D,D,D,W,W]','[D,D,D,W,D,D,D,W]'])if(!chapter.includes(fact))throw Error('Missing source fact: '+fact);
if(!originals[1].includes('![flame graph](./images/flamegraph.svg)')||
 !originals[2].includes('Copyright (c) 2018-2019 Ben Frederickson'))throw Error('Provenance missing');
console.log('PASS three pinned py-spy source/license files, complete original SVG,source sample counts,ancestry,percentages and order/double-count counterexamples'
 +(process.argv.includes('--upstream')?' against upstream':'')+'; no live sampling or GPU claim');
