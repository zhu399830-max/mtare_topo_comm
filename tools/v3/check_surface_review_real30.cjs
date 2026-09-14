// Actual Chrome file-input/UI test with synthetic data, no dataset access.
const test=require('node:test'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process');const path=require('node:path');
const {mkdtempSync,rmSync,readFileSync,writeFileSync}=require('node:fs');const {createHash}=require('node:crypto');const os=require('node:os');
test('Sealed30 review files render without submitting annotations', {timeout:120000},async()=>{
 const profile=mkdtempSync(path.join(os.tmpdir(),'gse-review-chrome-'));
 const chrome=spawn('/usr/bin/google-chrome',['--headless','--no-sandbox','--disable-gpu','--remote-debugging-pipe',`--user-data-dir=${profile}`],{stdio:['ignore','ignore','ignore','pipe','pipe']});
 let buffer='',seq=0;const pending=new Map();
 chrome.stdio[4].on('data',data=>{buffer+=data;let end;while((end=buffer.indexOf('\0'))>=0){const m=JSON.parse(buffer.slice(0,end));buffer=buffer.slice(end+1);const p=pending.get(m.id);if(p){pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}}});
 const send=(method,params={},sessionId)=>new Promise((resolve,reject)=>{const id=++seq;const timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout '+method));},8000);pending.set(id,{resolve,reject,timer});chrome.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+'\0');});
 try{
  const {targetId}=await send('Target.createTarget',{url:'about:blank'});
  const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});
  await send('Emulation.setDeviceMetricsOverride',{width:1500,height:1300,deviceScaleFactor:1,mobile:false},sessionId);
  const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true},sessionId);if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  const wait=async expr=>{for(let i=0;i<100;i++){if(await evaluate(expr))return;await new Promise(r=>setTimeout(r,20));}throw Error('UI wait '+expr);};
  const root=path.resolve(__dirname,'../..');
  const run=path.join(root,'results/gate3_semantics/gate3_20260907_gse_surface_review_export_v1_seed20260906');
  const hash=b=>createHash('sha256').update(b).digest('hex');
  const seal=readFileSync(path.join(run,'artifacts/evidence_sha256.txt'));
  assert.equal(hash(seal),'8cd3037b74c5c0fac6b4b8e92dadda1121660ab3c679c249035c84c0e4d26cb5');
  const pinned=new Map(seal.toString().trim().split('\n').map(l=>[l.slice(66),l.slice(0,64)]));
  const read=relative=>{const p=path.join(run,relative),rel=path.relative(root,p);assert.ok(pinned.has(rel));const raw=readFileSync(p);assert.equal(hash(raw),pinned.get(rel));return raw;};
  read('gse_surface_review.html');read('gse_surface_review.js');
  const manifest=JSON.parse(read('artifacts/review_manifest.json'));assert.equal(manifest.observations.length,30);
  for(let i=0;i<manifest.observations.length;i++){
    const row=manifest.observations[i],raw=read(row.blind),b=JSON.parse(raw);
    assert.equal(b.points_xyz_m.length,row.valid_points);
    await send('Page.navigate',{url:'file://'+path.join(run,'gse_surface_review.html')+'?observation='+i},sessionId);
    await wait('typeof SurfaceBrowserReview === "function"');
    await evaluate("document.getElementById('reviewer').value='READ_ONLY_RENDER_CHECK_NOT_ANNOTATION'");
    const doc=await send('DOM.getDocument',{},sessionId);
    const node=await send('DOM.querySelector',{nodeId:doc.root.nodeId,selector:'#bundle'},sessionId);
    await send('DOM.setFileInputFiles',{nodeId:node.nodeId,files:[path.join(run,row.blind)]},sessionId);
    await wait("document.getElementById('bundle').disabled");
    assert.equal(await evaluate("document.getElementById('reference').disabled"),true);
    const pixels=await evaluate("(()=>{const c=document.getElementById('three'),p=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let j=0;j<p.length;j+=4)if(p[j+3]===255&&!(p[j]===255&&p[j+1]===255&&p[j+2]===255))n++;return n;})()");
    assert.ok(pixels>0,'actual colored point pixels');
    const full=await evaluate("document.getElementById('three').toDataURL()");
    await evaluate("document.getElementById('history').value=0;document.getElementById('history').dispatchEvent(new Event('input'))");
    const prefix=await evaluate("document.getElementById('three').toDataURL()");
    assert.notEqual(full,prefix,'history prefix changes image');
    assert.equal(await evaluate("document.getElementById('reference').disabled && !document.getElementById('annotation').disabled"),true);
    if(i===0){await evaluate("document.getElementById('history').value=4;document.getElementById('history').dispatchEvent(new Event('input'))");const shot=await send('Page.captureScreenshot',{format:'png'},sessionId);writeFileSync('/tmp/gse_review_real30_first.png',Buffer.from(shot.data,'base64'));}
    console.log(JSON.stringify({checked:i+1,view_id:row.view_id,point_pixels:pixels,annotations_submitted:0}));
  }
 } finally {
  chrome.kill('SIGTERM');await new Promise(resolve=>chrome.once('exit',resolve));
  for(const p of pending.values())clearTimeout(p.timer);
  rmSync(profile,{recursive:true,force:true}); // exact test-owned mkdtemp directory
 }
});
