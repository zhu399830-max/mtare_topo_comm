// Actual Chrome file-input/UI test with synthetic data, no dataset access.
const test=require('node:test'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process');const path=require('node:path');
const {mkdtempSync,rmSync}=require('node:fs');const os=require('node:os');
test('Chrome loads five-frame file, locks judgment, validates reference, exports', {timeout:20000},async()=>{
 const profile=mkdtempSync(path.join(os.tmpdir(),'gse-review-chrome-'));
 const chrome=spawn('/usr/bin/google-chrome',['--headless','--no-sandbox','--disable-gpu','--remote-debugging-pipe',`--user-data-dir=${profile}`],{stdio:['ignore','ignore','ignore','pipe','pipe']});
 let buffer='',seq=0;const pending=new Map();
 chrome.stdio[4].on('data',data=>{buffer+=data;let end;while((end=buffer.indexOf('\0'))>=0){const m=JSON.parse(buffer.slice(0,end));buffer=buffer.slice(end+1);const p=pending.get(m.id);if(p){pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}}});
 const send=(method,params={},sessionId)=>new Promise((resolve,reject)=>{const id=++seq;const timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout '+method));},8000);pending.set(id,{resolve,reject,timer});chrome.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+'\0');});
 try{
  const {targetId}=await send('Target.createTarget',{url:'about:blank'});
  const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});
  const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true},sessionId);if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  const wait=async expr=>{for(let i=0;i<100;i++){if(await evaluate(expr))return;await new Promise(r=>setTimeout(r,20));}throw Error('UI wait '+expr);};
  await send('Page.navigate',{url:'file://'+path.resolve(__dirname,'../../../tools/v3/review/gse_surface_review.html')},sessionId);
  await wait('typeof SurfaceBrowserReview === "function"');
  await evaluate(`window.syntheticBundle={schema:'gse_surface_review_bundle_v1',observation_id:'synthetic-only',coordinate_frame:'current_sensor_m',source_frame_indices:[1,2,3,4,5],points_xyz_m:[[1,1,1],[-1,1,1],[1,-1,1],[-1,-1,1],[2,2,2]],point_history_slots:[0,1,2,3,4]};
    window.loadFile=(id,value)=>{const d=new DataTransfer();d.items.add(new File([JSON.stringify(value)],'synthetic.json',{type:'application/json'}));document.getElementById(id).files=d.files;document.getElementById(id).dispatchEvent(new Event('change'));};
    document.getElementById('reviewer').value='SYNTHETIC_AUTOMATION_NOT_HUMAN';loadFile('bundle',syntheticBundle);`);
  await wait("document.getElementById('bundle').disabled");
  // The legacy bundle must not silently invent raw identities for region export.
  await evaluate("document.getElementById('regionSpec').value=JSON.stringify({bounds_m:[[-3,-3,-3],[3,3,3]],history_slots:[0,1,2,3,4],kind:'surface_region',evidence:'synthetic'});document.getElementById('regionSelect').click()");
  assert.match(await evaluate("document.getElementById('status').textContent"),/v2/);
  assert.equal(await evaluate("document.getElementById('reference').disabled"),true);
  assert.equal(await evaluate("(()=>{const c=document.getElementById('three'),p=c.getContext('2d').getImageData(0,0,c.width,c.height).data;for(let i=0;i<p.length;i+=4)if(p[i]===34&&p[i+1]===197&&p[i+2]===94&&p[i+3]===255)return true;return false;})()"),true);
  await evaluate("document.getElementById('opening').click()");
  assert.equal(await evaluate("JSON.parse(document.getElementById('annotation').value).anchors.length"),0);
  await evaluate(`let a=JSON.parse(document.getElementById('annotation').value);a.openings[0].position_m=[2,0,0];a.openings[0].evidence='synthetic UI test';document.getElementById('annotation').value=JSON.stringify(a);document.getElementById('commit').click();`);
  assert.equal(await evaluate("document.getElementById('annotation').disabled"),true);
  // Intercept only browser download creation, retaining the production export.
  await evaluate("window.savedBlob=null;URL.createObjectURL=b=>{window.savedBlob=b;return 'blob:synthetic';};HTMLAnchorElement.prototype.click=function(){};document.getElementById('save').click()");
  const partial=await evaluate('savedBlob.text().then(JSON.parse)');
  assert.equal(partial.blind_annotation.openings.length,1);
  await evaluate(`loadFile('reference',{bundle_file_sha256:${JSON.stringify(partial.bundle_file_sha256)},construction_reference:{note:'synthetic-only'}})`);
  await wait("!document.getElementById('finish').disabled");
  await evaluate("document.getElementById('notes').value='synthetic, not human review';document.getElementById('finish').click();document.getElementById('save').click()");
  const out=await evaluate('savedBlob.text().then(JSON.parse)');
  assert.equal(out.comparison_notes,'synthetic, not human review');assert.equal(out.automatic_training_eligibility,false);
  assert.match(out.reference_file_sha256,/^[a-f0-9]{64}$/);
  await send('Page.navigate',{url:'file://'+path.resolve(__dirname,'../../../tools/v3/review/gse_surface_review.html')+'?v2'},sessionId);
  await wait('typeof SurfaceBrowserReview === "function"');
  await evaluate(`const b={schema:'gse_surface_review_bundle_v2',observation_id:'synthetic-v2',coordinate_frame:'current_sensor_m',source_frame_indices:[1,2,3,4,5],points_xyz_m:[[1,1,1],[1,1,8]],point_history_slots:[0,4],point_ray_indices:[721,722]};
    const d=new DataTransfer();d.items.add(new File([JSON.stringify(b)],'synthetic-v2.json',{type:'application/json'}));document.getElementById('reviewer').value='AI_SYNTHETIC';document.getElementById('bundle').files=d.files;document.getElementById('bundle').dispatchEvent(new Event('change'));`);
  await wait("document.getElementById('bundle').disabled");
  await evaluate("document.getElementById('regionSpec').value=JSON.stringify({bounds_m:[[0,0,0],[2,2,2]],history_slots:[0,4],kind:'surface_region',evidence:'synthetic height separation'});document.getElementById('regionSelect').click();window.savedBlob=null;URL.createObjectURL=b=>{window.savedBlob=b;return 'blob:synthetic';};HTMLAnchorElement.prototype.click=function(){};document.getElementById('regionSave').click()");
  const region=await evaluate('savedBlob.text().then(JSON.parse)');
  assert.equal(region.blind,false);assert.equal(region.prior_reference_exposure,true);
  assert.equal(region.regions[0].points.length,1);assert.equal(region.regions[0].points[0].ring,1);
  assert.equal(region.regions[0].points[0].azimuth_bin,1);assert.equal(region.regions[0].structural_membership,null);
  assert.equal(region.automatic_training_eligibility,false);
 } finally {
  chrome.kill('SIGTERM');await new Promise(resolve=>chrome.once('exit',resolve));
  for(const p of pending.values())clearTimeout(p.timer);
  try {rmSync(profile,{recursive:true,force:true,maxRetries:3,retryDelay:100});}
  catch(e){if(e.code!=='ENOTEMPTY')throw e;console.warn('Chrome child still writing; retained synthetic test profile:',profile);} // do not mask assertion failures with cleanup races
 }
});
