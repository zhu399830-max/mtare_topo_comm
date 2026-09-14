// Browser-only QA of the generated report. No source experiments are executed.
const {spawn}=require('node:child_process');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const assert=require('node:assert/strict');
(async()=>{
 const output=fs.mkdtempSync(path.join(os.tmpdir(),'gse-report-review-'));
 const chrome=spawn('/usr/bin/google-chrome',['--headless','--no-sandbox','--disable-gpu','--remote-debugging-pipe',`--user-data-dir=${output}/profile`],{stdio:['ignore','ignore','ignore','pipe','pipe']});
 let buffer='',seq=0;const pending=new Map();
 chrome.stdio[4].on('data',data=>{buffer+=data;let end;while((end=buffer.indexOf('\0'))>=0){const m=JSON.parse(buffer.slice(0,end));buffer=buffer.slice(end+1);const p=pending.get(m.id);if(p){pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}}});
 const send=(method,params={},sessionId)=>new Promise((resolve,reject)=>{const id=++seq;const timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout '+method));},20000);pending.set(id,{resolve,reject,timer});chrome.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+'\0');});
 try{
  const {targetId}=await send('Target.createTarget',{url:'about:blank'});
  const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});
  const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true},sessionId);if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false},sessionId);
  const report=path.resolve(__dirname,'../../docs/GSE_RESEARCH_REVIEW_20260914.html');
  await send('Page.navigate',{url:'file://'+report},sessionId);
  for(let i=0;i<100;i++){if(await evaluate('document.readyState === "complete"'))break;await new Promise(r=>setTimeout(r,50));}
  const result=await evaluate(`(async()=>{await document.fonts.ready;await Promise.all([...document.querySelectorAll('figure img')].map(i=>i.decode()));return {images:document.querySelectorAll('figure img').length,broken:[...document.querySelectorAll('figure img')].filter(i=>!i.naturalWidth).length,sections:document.querySelectorAll('main section').length,horizontalOverflow:document.documentElement.scrollWidth>innerWidth,missingAnchors:[...document.querySelectorAll('a[href^="#"]')].filter(a=>!document.querySelector(a.getAttribute('href'))).length,externalAssets:[...document.querySelectorAll('[src],link[href]')].filter(e=>/^(https?:)?\\/\\//.test(e.getAttribute('src')||e.getAttribute('href')||'')).length}})()`);
  assert.equal(result.broken,0);assert.equal(result.sections,12);assert.equal(result.missingAnchors,0);assert.equal(result.externalAssets,0);assert.equal(result.horizontalOverflow,false);
  await evaluate("document.documentElement.style.scrollBehavior='auto'");
  for(const [name,id] of [['cover',null],['detection','center'],['relations','branch'],['simulation','simulation']]){
   await evaluate(id?`document.getElementById('${id}').scrollIntoView()`:'scrollTo(0,0)');
   const shot=await send('Page.captureScreenshot',{format:'png'},sessionId);fs.writeFileSync(path.join(output,name+'.png'),Buffer.from(shot.data,'base64'));
  }
  await evaluate("document.querySelector('figure img').click()");assert.equal(await evaluate("document.querySelector('dialog').open"),true);await evaluate("document.querySelector('dialog').close()");
  await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true},sessionId);
  await evaluate('scrollTo(0,0)');const overflow=await evaluate('document.documentElement.scrollWidth>innerWidth');assert.equal(overflow,false);
  const shot=await send('Page.captureScreenshot',{format:'png'},sessionId);fs.writeFileSync(path.join(output,'mobile.png'),Buffer.from(shot.data,'base64'));
  console.log(JSON.stringify({...result,mobileOverflow:overflow,zoomWorks:true,screenshots:output}));
 }finally{chrome.kill();}
})().catch(e=>{console.error(e);process.exitCode=1;});
