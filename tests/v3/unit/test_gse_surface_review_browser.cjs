const test=require('node:test'),assert=require('node:assert/strict');
const {SurfaceBrowserReview,emptyTarget,parseStrictJSON}=require('../../../tools/v3/review/gse_surface_review.js');
const b=()=>({schema:'gse_surface_review_bundle_v1',observation_id:'opaque',coordinate_frame:'current_sensor_m',source_frame_indices:[1,2,3,4,5],points_xyz_m:[[1,2,3]],point_history_slots:[4]});
test('one observation and independent opening, locked before reference',()=>{
 const bundle=b(),s=new SurfaceBrowserReview(bundle,'a'.repeat(64),'synthetic');
 assert.throws(()=>s.reveal({},'b'.repeat(64)));
 const a=emptyTarget(bundle);a.openings=[{position_m:[1,0,0],direction:null,width_m:null,height_m:null,evidence:'synthetic'}];a.membership=[[]];s.commit(a);
 a.openings=[];assert.equal(s.export().blind_annotation.openings.length,1);
 assert.throws(()=>s.commit(emptyTarget(bundle)));
 s.reveal({bundle_file_sha256:'a'.repeat(64),construction_reference:{}},'b'.repeat(64));s.finish('synthetic comparison');
 assert.equal(s.export().automatic_training_eligibility,false);
});
test('unknown masks not turned into completeness',()=>{
 const s=new SurfaceBrowserReview(b(),'a'.repeat(64),'synthetic');s.commit(emptyTarget(b()));
 assert.equal(s.export().blind_annotation.score_region.anchors_complete,false);
});
test('duplicate JSON keys and wrong frames rejected',()=>{
 assert.throws(()=>parseStrictJSON('{"x":1,"x":2}'));
 const v=b();v.source_frame_indices=[1,3,2,4,5];assert.throws(()=>new SurfaceBrowserReview(v,'a'.repeat(64),'test'));
});
test('teacher fields rejected in blind input',()=>{const v=b();v.node_id='secret';assert.throws(()=>new SurfaceBrowserReview(v,'a'.repeat(64),'test'));});
