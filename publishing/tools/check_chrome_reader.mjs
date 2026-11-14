/** Inspect the local built reader in an isolated, already-running Chrome.
 * node publishing/tools/check_chrome_reader.mjs ROUTE OUTPUT_DIR [WIDTH] [REFERENCE_ANCHOR]
 * Start Chrome with a separate temporary profile and debugging port9341.
 * Does not read cookies, user profiles or external pages. Screenshots and
 * geometry are evidence for manual review, not automatic visual acceptance.
 */
import {mkdir,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';

const [route,outArg,widthArg='1280',referenceAnchor]=process.argv.slice(2);
const width=Number(widthArg),height=width<700?812:900;
if(!route?.startsWith('/')||route.startsWith('//')||!outArg||![375,1280].includes(width))
  throw Error('Require a local route, output directory and width375 or1280');
const url=new URL(route,'http://127.0.0.1:4330');
if(url.origin!=='http://127.0.0.1:4330')throw Error('Local preview only');
const tabs=await(await fetch('http://127.0.0.1:9341/json/list')).json();
const target=tabs.find(t=>t.type==='page'&&(t.url==='about:blank'||t.url.startsWith(url.origin)));
if(!target)throw Error('No isolated local-reader tab');
const ws=new WebSocket(target.webSocketDebuggerUrl),pending=new Map();
let sequence=0;
ws.addEventListener('message',event=>{
  const m=JSON.parse(event.data),entry=pending.get(m.id);
  if(!entry)return;
  clearTimeout(entry.timer);pending.delete(m.id);
  m.error?entry.reject(Error(JSON.stringify(m.error))):entry.resolve(m.result);
});
await new Promise((res,rej)=>{ws.addEventListener('open',res,{once:true});ws.addEventListener('error',rej,{once:true});});
const send=(method,params={})=>new Promise((res,rej)=>{
  const id=++sequence,timer=setTimeout(()=>{pending.delete(id);rej(Error('CDP timeout: '+method));},20000);
  pending.set(id,{resolve:res,reject:rej,timer});ws.send(JSON.stringify({id,method,params}));
});
const evaluate=async expression=>{
  const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
  if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
};
const wait=async expression=>{
  const deadline=Date.now()+20000;let error;
  while(Date.now()<deadline){
    try{if(await evaluate(expression))return;}catch(e){error=e;}
    await new Promise(r=>setTimeout(r,100));
  }
  throw Error('Reader condition timed out: '+expression+' '+(error?.message||''));
};
const click=async selector=>{
  const box=await evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e)throw Error('missing click target');e.scrollIntoView({block:'center',behavior:'instant'});const b=e.getClientRects()[0]||e.getBoundingClientRect();return{x:b.x+b.width/2,y:b.y+b.height/2}})()`);
  await send('Input.dispatchMouseEvent',{type:'mouseMoved',...box});
  await send('Input.dispatchMouseEvent',{type:'mousePressed',...box,button:'left',buttons:1,clickCount:1});
  await send('Input.dispatchMouseEvent',{type:'mouseReleased',...box,button:'left',buttons:0,clickCount:1});
};
const out=resolve(outArg);await mkdir(out,{recursive:true});
const capture=async name=>{
  const r=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  const path=resolve(out,width+'-'+name+'.png');await writeFile(path,Buffer.from(r.data,'base64'));return path;
};
try{
  await send('Page.enable');await send('Runtime.enable');await send('Page.bringToFront');
  await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width<700});
  let navigation;
  if(referenceAnchor){
    const from=(url.pathname.startsWith('/en/')?'/en':'')+'/reference/';
    await send('Page.navigate',{url:url.origin+from});
    await wait(`location.pathname===${JSON.stringify(from)}&&document.readyState==='complete'&&!!document.querySelector('.sl-markdown-content')`);
    const selected=await evaluate(`(()=>{const a=Array.from(document.querySelectorAll('.sl-markdown-content a')).find(a=>new URL(a.href).pathname===${JSON.stringify(url.pathname)}&&new URL(a.href).hash===${JSON.stringify('#'+referenceAnchor)});if(!a)throw Error('missing subject link');a.dataset.readerQaNav='target';return{href:a.href,text:a.textContent}})()`);
    await click('[data-reader-qa-nav="target"]');
    await wait(`location.pathname===${JSON.stringify(url.pathname)}&&location.hash===${JSON.stringify('#'+referenceAnchor)}&&!!document.getElementById(${JSON.stringify(referenceAnchor)})`);
    navigation={from,...selected,arrived:await evaluate('location.href'),screenshot:await capture('reference-target')};
  }else await send('Page.navigate',{url:url.href});
  await wait(`location.pathname===${JSON.stringify(url.pathname)}&&document.readyState==='complete'&&!!document.querySelector('.sl-markdown-content')`);
  await evaluate('document.fonts.ready.then(()=>true)');
  const images=referenceAnchor?[]:await evaluate(`Array.from(document.querySelectorAll('.sl-markdown-content a>img')).map((e,i)=>{e.dataset.readerQa=String(i);return{index:i,src:e.currentSrc||e.src,alt:e.alt}})`);
  const report={url:url.href,width,height,navigation,figures:[],limits:'Headless browser interactions and screenshots; requires separate human visual inspection.'};
  for(const info of images){
    const selector='img[data-reader-qa="'+info.index+'"]';
    await evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({block:'center',behavior:'instant'})`);
    await wait(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});return e.complete&&e.naturalWidth>0})()`);
    await click(selector);
    await wait("document.querySelector('[data-figure-viewer]').open&&document.querySelector('[data-viewer-image]').naturalWidth>0&&getComputedStyle(document.querySelector('[data-viewer-image]')).display!=='none'");
    const state=await evaluate(`(()=>{const e=document.querySelector('[data-viewer-image]'),v=document.querySelector('[data-viewer-viewport]'),b=e.getBoundingClientRect();return{natural:[e.naturalWidth,e.naturalHeight],display:[b.width,b.height],viewport:[v.clientWidth,v.clientHeight],scroll:[v.scrollWidth,v.scrollHeight],caption:document.querySelector('[data-viewer-caption]').textContent.trim(),scale:document.querySelector('[data-viewer-scale]').textContent}})()`);
    if(!state.caption)throw Error('Missing figure attribution: '+info.src);
    const screenshot=await capture('figure-'+String(info.index).padStart(2,'0'));
    const inspected={...info,...state,screenshot};
    if(width<700){
      // Opening/closing use native pointer events. Zoom exercises the actual
      // six button handlers in one evaluation, without six compositor waits.
      await evaluate("(()=>{for(let n=0;n<6;n++)document.querySelector('[data-viewer-plus]').click();return true})()");
      inspected.zoom=await evaluate(`(()=>{const v=document.querySelector('[data-viewer-viewport]');v.scrollTo({left:0,top:0,behavior:'instant'});return{scale:document.querySelector('[data-viewer-scale]').textContent,atStart:[v.scrollLeft,v.scrollTop],extent:[v.scrollWidth-v.clientWidth,v.scrollHeight-v.clientHeight]}})()`);
      inspected.zoom.startScreenshot=await capture('figure-'+String(info.index).padStart(2,'0')+'-zoom-start');
      inspected.zoom.atEnd=await evaluate(`(()=>{const v=document.querySelector('[data-viewer-viewport]');v.scrollTo({left:v.scrollWidth,top:v.scrollHeight,behavior:'instant'});return[v.scrollLeft,v.scrollTop]})()`);
      inspected.zoom.endScreenshot=await capture('figure-'+String(info.index).padStart(2,'0')+'-zoom-end');
    }
    report.figures.push(inspected);
    await click('[data-viewer-close]');
    await wait("!document.querySelector('[data-figure-viewer]').open");
    console.log(JSON.stringify({width,figure:info.index+1,total:images.length,status:'captured'}));
  }
  report.layout=await evaluate(`(()=>{window.scrollTo(0,0);const m=document.querySelector('.sl-markdown-content');return{page:[document.documentElement.clientWidth,document.documentElement.scrollWidth],mathErrors:m.querySelectorAll('.katex-error').length,formulas:Array.from(m.querySelectorAll('.katex-display')).map(e=>[e.clientWidth,e.scrollWidth]),tables:Array.from(m.querySelectorAll('table')).map(e=>{const p=e.parentElement;return[p.clientWidth,p.scrollWidth,getComputedStyle(p).overflowX]})}})()`);
  report.contextScreenshot=await capture('page');
  await writeFile(resolve(out,width+'-report.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({url:url.href,width,figures:images.length,layout:report.layout,report:resolve(out,width+'-report.json')}));
}finally{ws.close();}
