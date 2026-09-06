const {test}=require('node:test');
const assert=require('node:assert/strict');
const crypto=require('node:crypto');
const {BrowserReview,validateBundle,validateAnnotation,emptyAnnotation,parseStrictJSON,MAX_REVIEW_FILE_BYTES}=require('../../../tools/v3/review/gse_structure_review.js');
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
function sorted(x){if(Array.isArray(x))return x.map(sorted);if(x&&typeof x==='object')return Object.fromEntries(Object.keys(x).sort().map(k=>[k,sorted(x[k])]));return x;}
function bundle(){return {schema:'gse_structure_blind_bundle_v1',bundle_id:'anonymous',coordinate_frame:'current_sensor_m',decisions:Array.from({length:21},(_,i)=>({decision_index:i,source_frame_keys:Array.from({length:5},(_,j)=>'f'+(i+j)),source_order_indices:Array.from({length:5},(_,j)=>i+j),points_xyz_m:[[0,0,1]]}))};}
const fileSha=b=>sha(JSON.stringify(b));
function reference(b){return {schema:'gse_structure_reference_v1',bundle_id:b.bundle_id,
  // Integer-only fixture, not an implementation of Python float JSON.
  blind_bundle_sha256:sha(JSON.stringify(sorted(b))),blind_bundle_file_sha256:fileSha(b),
  decisions:b.decisions.map(d=>({decision_index:d.decision_index,reference_annotation:emptyAnnotation()}))};}
function complete(b){const s=new BrowserReview(b,fileSha(b),'reviewer');for(let i=0;i<21;i++)s.commit(i,emptyAnnotation());return s;}
test('blind-before-reference and no backfill',()=>{const b=bundle(),s=new BrowserReview(b,fileSha(b),'reviewer'),r=reference(b),h=sha(JSON.stringify(r));assert.throws(()=>s.reveal(r,h));for(let i=0;i<21;i++)s.commit(i,emptyAnnotation());const before=s.export().blind_records;s.reveal(r,h);s.finish('checked');assert.deepEqual(s.export().blind_records,before);assert.throws(()=>s.commit(0,emptyAnnotation()));assert.throws(()=>s.finish('edit'));assert.equal(s.export().automatic_training_eligibility,false);});
test('source reverse and identity drift rejected',()=>{const b=bundle();b.decisions[2].source_order_indices=[0,1,2,3,4];assert.throws(()=>validateBundle(b));const c=bundle();c.decisions[1].source_frame_keys[0]='other';assert.throws(()=>validateBundle(c));});
test('bad annotation never locks a decision',()=>{const b=bundle(),s=new BrowserReview(b,fileSha(b),'reviewer');const a=emptyAnnotation();a.complete_regions=[{min_xyz_m:[null,null,null],max_xyz_m:[null,null,null],reason:'template'}];assert.throws(()=>s.commit(0,a));assert.equal(s.current().decision_index,0);});
test('unknown dimensions stay null',()=>{const a=emptyAnnotation();a.structures=[{local_id:0,event:'junction',center_xyz_m:null,evidence_note:'visible',openings:[{position_xyz_m:null,direction:null,width_m:null,height_m:null,evidence_note:'visible direction'}]}];validateAnnotation(a);assert.equal(a.structures[0].openings[0].width_m,null);});
test('caller cannot mutate committed observations',()=>{const b=bundle(),s=new BrowserReview(b,fileSha(b),'reviewer');b.decisions[0].points_xyz_m[0][2]=999;assert.equal(s.current().points_xyz_m[0][2],1);const a=emptyAnnotation();s.commit(0,a);a.notes='future';assert.equal(s.export().blind_records[0].annotation.notes,'');});
test('wrong-content reference cannot be exposed with same opaque bundle id',()=>{const b=bundle(),s=complete(b),other=bundle();other.decisions[0].points_xyz_m[0][0]=999;const r=reference(other);assert.throws(()=>s.reveal(r,sha(JSON.stringify(r))));assert.equal(s.reference,null);assert.equal(s.referenceSha,null);assert.throws(()=>s.finish('looked at wrong reference'));});
test('malformed reference rows/annotations do not lock reveal state',()=>{for(const mutate of [r=>delete r.blind_bundle_file_sha256,r=>{r.decisions[0].extra='teacher';},r=>{r.decisions[0].reference_annotation.structures='bad';}]){const b=bundle(),s=complete(b),r=reference(b);mutate(r);assert.throws(()=>s.reveal(r,sha(JSON.stringify(r))));assert.equal(s.reference,null);assert.equal(s.referenceSha,null);}});
test('shared per-file cap is 128 MiB, not unlimited decoded input',()=>assert.equal(MAX_REVIEW_FILE_BYTES,128*1024*1024));
test('strict raw JSON rejects root and nested duplicate keys before exposure',()=>{
  for(const raw of ['{"a":1,"a":2}','{"outer":[{"a":1,"a":2}]}','{"a":{"z":0,"z":1}}'])assert.throws(()=>parseStrictJSON(raw),/重复JSON/);
});
test('decoded escaped names count as the same object key',()=>{
  for(const raw of [String.raw`{"a":1,"\u0061":2}`,String.raw`{"x\\y":0,"x\u005cy":1}`,String.raw`{"\"":0,"\u0022":1}`])assert.throws(()=>parseStrictJSON(raw),/重复JSON/);
});
test('same names in independent objects are legal, string punctuation is not structure',()=>{
  const raw=String.raw`{"rows":[{"a":1},{"a":2}],"other":{"a":"{ \"a\": 1, \"a\": 2 }"},"array":["a","a"],"empty":{}}`;
  assert.deepEqual(parseStrictJSON(raw),JSON.parse(raw));
});
test('strict scanner retains original JSON syntax errors and non-object roots',()=>{
  for(const raw of ['{"a":1,}','{"a" 1}','[1,]','{"a":NaN}'])assert.throws(()=>parseStrictJSON(raw),SyntaxError);
  for(const raw of ['null','4.25','"scalar"','[1,2]'])assert.deepEqual(parseStrictJSON(raw),JSON.parse(raw));
  assert.throws(()=>parseStrictJSON({}),/文本/);
});
