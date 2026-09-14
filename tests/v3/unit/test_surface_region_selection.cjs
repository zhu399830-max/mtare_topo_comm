const {test}=require('node:test'),assert=require('node:assert/strict');
const {selectSurfaceRegion,validateBundle}=require('../../../tools/v3/review/gse_surface_review.js');
const bundle=()=>({schema:'gse_surface_review_bundle_v2',observation_id:'synthetic',coordinate_frame:'current_sensor_m',source_frame_indices:[10,11,12,13,14],points_xyz_m:[[1,2,3],[1,2,8],[1,2,3],[20,2,3]],point_history_slots:[0,0,4,4],point_ray_indices:[721,722,721,723]});
test('3D height and frame selection preserve exact raw ray identity',()=>{
 const b=bundle(),r=selectSurfaceRegion(b,[[0,0,0],[2,3,4]],[0],'surface_region','synthetic selection');
 assert.deepEqual(r.points,[{frame_row:10,ring:1,azimuth_bin:1,position_m:[1,2,3]}]);
 assert.equal(r.structural_membership,null);assert.equal(r.training_qualified,false);assert.equal(r.physical_passability,null);
 assert.equal(selectSurfaceRegion(b,[[0,0,0],[30,3,9]],[4],'surface_region','test').points.length,1);
});
test('query order does not change selected physical identities',()=>{
 const b=bundle(),c=structuredClone(b);for(const k of ['points_xyz_m','point_history_slots','point_ray_indices'])c[k].reverse();
 const pick=x=>selectSurfaceRegion(x,[[0,0,0],[2,3,4]],[0,4],'surface_region','test').points.sort((a,b)=>a.frame_row-b.frame_row);
 assert.deepEqual(pick(b),pick(c));
});
test('missing, duplicate and invalid source indices are rejected',()=>{
 const b=bundle();b.point_ray_indices[1]=721;assert.throws(()=>validateBundle(b));
 b.point_ray_indices[1]=11520;assert.throws(()=>validateBundle(b));
 delete b.point_ray_indices;assert.throws(()=>validateBundle(b));
});
test('no empty annotation, invalid bounds or automatic structure label',()=>{
 const b=bundle();assert.throws(()=>selectSurfaceRegion(b,[[9,9,9],[10,10,10]],[0],'surface_region','test'));
 assert.throws(()=>selectSurfaceRegion(b,[[2,0,0],[1,3,4]],[0],'surface_region','test'));
 assert.throws(()=>selectSurfaceRegion(b,[[0,0,0],[2,3,4]],[0],'confirmed_junction','test'));
 const old=structuredClone(b);old.schema='gse_surface_review_bundle_v1';delete old.point_ray_indices;validateBundle(old);
 assert.throws(()=>selectSurfaceRegion(old,[[0,0,0],[2,3,4]],[0],'surface_region','test'));
});
