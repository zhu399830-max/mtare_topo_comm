"use strict";
const clone=x=>JSON.parse(JSON.stringify(x));
const keys=(x,ks)=>{if(!x||Array.isArray(x)||typeof x!=="object"||Object.keys(x).sort().join()!==ks.split(" ").sort().join())throw Error("字段缺失或包含额外字段");};
const xyz=x=>{if(!Array.isArray(x)||x.length!==3||x.some(v=>typeof v!=="number"||!Number.isFinite(v)))throw Error("需要有限三维坐标");};
const text=x=>{if(typeof x!=="string"||!x.trim())throw Error("需要填写依据");};
const digest=x=>typeof x==="string"&&/^[a-f0-9]{64}$/.test(x);
function parseStrictJSON(raw){
  const value=JSON.parse(raw),stack=[];const re=/"(?:\\.|[^"\\])*"|[{}\[\],:]/g;let m;
  while((m=re.exec(raw))){const t=m[0],top=stack.at(-1);if(t==="{")stack.push({keys:new Set(),key:true});else if(t==="[")stack.push({});else if(t==="}"||t==="]")stack.pop();else if(t===","&&top?.keys)top.key=true;else if(t[0]==='"'&&top?.keys&&top.key){const k=JSON.parse(t);if(top.keys.has(k))throw Error("重复字段");top.keys.add(k);top.key=false;}}
  return value;
}
function validateBundle(b){
  const indexed=b.schema==="gse_surface_review_bundle_v2";
  keys(b,"schema observation_id coordinate_frame source_frame_indices points_xyz_m point_history_slots"+(indexed?" point_ray_indices":""));
  if(!["gse_surface_review_bundle_v1","gse_surface_review_bundle_v2"].includes(b.schema)||b.coordinate_frame!=="current_sensor_m")throw Error("错误的单观察格式");text(b.observation_id);
  const f=b.source_frame_indices;if(!Array.isArray(f)||f.length!==5||f.some((x,i)=>!Number.isSafeInteger(x)||x<0||(i&&x<=f[i-1])))throw Error("需要五个递增源帧");
  if(!Array.isArray(b.points_xyz_m)||b.points_xyz_m.length>57600||!Array.isArray(b.point_history_slots)||b.points_xyz_m.length!==b.point_history_slots.length)throw Error("点数或历史归属错误");
  b.points_xyz_m.forEach(xyz);if(b.point_history_slots.some(s=>!Number.isInteger(s)||s<0||s>4))throw Error("历史帧超范围");
  if(indexed){
    if(!Array.isArray(b.point_ray_indices)||b.point_ray_indices.length!==b.points_xyz_m.length)throw Error("原射线索引缺失");
    const seen=new Set();b.point_ray_indices.forEach((ray,i)=>{const key=b.point_history_slots[i]+":"+ray;if(!Number.isInteger(ray)||ray<0||ray>=11520||seen.has(key))throw Error("原射线索引重复或越界");seen.add(key);});
  }
}
function selectSurfaceRegion(b,bounds,slots,kind,evidence){
  validateBundle(b);if(b.schema!=="gse_surface_review_bundle_v2")throw Error("必须使用含原射线索引的v2观察包");
  if(!Array.isArray(bounds)||bounds.length!==2)throw Error("需要三维最小/最大边界");bounds.forEach(xyz);
  if(bounds[0].some((v,i)=>v>bounds[1][i]))throw Error("边界顺序错误");
  if(!Array.isArray(slots)||!slots.length||new Set(slots).size!==slots.length||slots.some(s=>!Number.isInteger(s)||s<0||s>4))throw Error("显式选择历史帧");
  if(!["surface_region","opening_boundary_candidate","termination_surface_candidate","occlusion_boundary_candidate"].includes(kind))throw Error("区域类型不支持");text(evidence);
  const points=[];b.points_xyz_m.forEach((p,i)=>{if(slots.includes(b.point_history_slots[i])&&Math.hypot(...p)<=10&&p.every((v,j)=>v>=bounds[0][j]&&v<=bounds[1][j])){const ray=b.point_ray_indices[i];points.push({frame_row:b.source_frame_indices[b.point_history_slots[i]],ring:Math.floor(ray/720),azimuth_bin:ray%720,position_m:[...p]});}});
  if(!points.length)throw Error("区域内没有有效返回，不能保存空证据");
  return {kind,bounds_m:clone(bounds),history_slots:[...slots],evidence,points,coordinate_source:"measured_surface",structural_membership:null,physical_passability:null,training_qualified:false};
}
function emptyTarget(b){return {schema:"gse_surface_observed_targets_v1",coordinate_frame:"current_sensor_m",source_frame_indices:[...b.source_frame_indices],anchors:[],openings:[],membership:[],score_region:{center_m:[0,0,0],radius_m:10,anchors_complete:false,openings_complete:false,evidence:"尚未确认完整；未标注不作负例"}};}
function validateTarget(a,b){
  keys(a,"schema coordinate_frame source_frame_indices anchors openings membership score_region");
  if(a.schema!=="gse_surface_observed_targets_v1"||a.coordinate_frame!=="current_sensor_m"||JSON.stringify(a.source_frame_indices)!==JSON.stringify(b.source_frame_indices))throw Error("标签源帧/格式不匹配");
  for(const [name,cap] of [["anchors",32],["openings",64]]){
    if(!Array.isArray(a[name])||a[name].length>cap)throw Error("容量超限，禁止截断");const seen=new Set();
    for(const x of a[name]){keys(x,name==="anchors"?"position_m evidence":"position_m direction width_m height_m evidence");xyz(x.position_m);text(x.evidence);if(seen.has(JSON.stringify(x.position_m)))throw Error("重复位置");seen.add(JSON.stringify(x.position_m));
      if(name==="openings"){if(x.direction!==null){xyz(x.direction);if(Math.abs(Math.hypot(...x.direction)-1)>1e-6)throw Error("方向需单位向量或null");}for(const k of ["width_m","height_m"])if(x[k]!==null&&(typeof x[k]!=="number"||!Number.isFinite(x[k])||x[k]<=0))throw Error("尺寸需正数或null");}}
  }
  if(!Array.isArray(a.membership)||a.membership.length!==a.openings.length||a.membership.some(r=>!Array.isArray(r)||r.length!==a.anchors.length||r.some(x=>x!==null&&typeof x!=="boolean")))throw Error("每个开口对应一行归属，true/false/null");
  const r=a.score_region;keys(r,"center_m radius_m anchors_complete openings_complete evidence");xyz(r.center_m);text(r.evidence);
  if(typeof r.radius_m!=="number"||!Number.isFinite(r.radius_m)||r.radius_m<=0||r.radius_m>10||typeof r.anchors_complete!=="boolean"||typeof r.openings_complete!=="boolean")throw Error("评分区域声明错误");
}
class SurfaceBrowserReview {
  constructor(bundle,fileSha,reviewer){validateBundle(bundle);if(!digest(fileSha))throw Error("需要源文件哈希");text(reviewer);this.bundle=clone(bundle);this.fileSha=fileSha;this.reviewer=reviewer;this.blind=null;this.reference=null;this.referenceSha=null;this.notes=null;}
  commit(a){if(this.blind)throw Error("盲看已锁定");validateTarget(a,this.bundle);this.blind=clone(a);}
  reveal(r,sha){if(!this.blind||this.reference)throw Error("先锁定盲看；参考只加载一次");keys(r,"bundle_file_sha256 construction_reference");if(r.bundle_file_sha256!==this.fileSha||!digest(sha))throw Error("参考绑定错误");this.reference=clone(r);this.referenceSha=sha;}
  finish(notes){if(!this.reference||this.notes!==null)throw Error("需要先显示参考，且只提交一次");text(notes);this.notes=notes;}
  export(){return clone({schema:"gse_surface_browser_review_v1",bundle_file_sha256:this.fileSha,reviewer_assertion:this.reviewer,blind_annotation:this.blind,reference_file_sha256:this.referenceSha,comparison_notes:this.notes,automatic_training_eligibility:false});}
}
if(typeof module!=="undefined")module.exports={SurfaceBrowserReview,validateBundle,validateTarget,emptyTarget,parseStrictJSON,selectSurfaceRegion};
if(typeof document!=="undefined"){
  const $=id=>document.getElementById(id);let session=null,yaw=.3,pitch=.45,zoom=1,drag=null;const surfaceRegions=[];
  const status=t=>$("status").textContent=t;
  const guard=fn=>async e=>{try{await fn(e);}catch(x){status("未完成："+x.message);}};
  const read=async f=>{if(!f||f.size>128*1024*1024)throw Error("文件上限128MiB");const buf=await f.arrayBuffer();return [parseStrictJSON(new TextDecoder().decode(buf)),Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",buf)),v=>v.toString(16).padStart(2,"0")).join("")];};
  const edit=()=>parseStrictJSON($("annotation").value);const set=a=>$("annotation").value=JSON.stringify(a,null,2);
  if($("regionSelect"))$("regionSelect").onclick=guard(()=>{
    if(!session||session.blind)throw Error("先载入v2观察包，锁定前才能新增区域");
    const r=parseStrictJSON($("regionSpec").value);keys(r,"bounds_m history_slots kind evidence");
    const selected=selectSurfaceRegion(session.bundle,r.bounds_m,r.history_slots,r.kind,r.evidence);
    surfaceRegions.push(selected);$("regionStatus").textContent=`已记录${surfaceRegions.length}个区域；本次${selected.points.length}个返回点。未确认结构身份。`;
  });
  if($("regionSave"))$("regionSave").onclick=guard(()=>{
    if(!session||!surfaceRegions.length)throw Error("尚无点级区域记录");
    const record={schema:"gse_surface_region_review_v1",bundle_file_sha256:session.fileSha,observation_id:session.bundle.observation_id,coordinate_frame:"current_sensor_m",reviewer_assertion:session.reviewer,prior_reference_exposure:true,blind:false,regions:clone(surfaceRegions),complete_background:false,automatic_training_eligibility:false};
    const u=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:"application/json"})),a=document.createElement("a");a.href=u;a.download="surface-regions.json";a.click();URL.revokeObjectURL(u);
  });
  function render(){if(!session)return;const b=session.bundle;const colors=["#6366f1","#22c55e","#eab308","#f97316","#22d3ee"];for(const kind of ["three","xy","xz"]){const c=$(kind),ctx=c.getContext("2d"),scale=Math.min(c.width,c.height)/22*zoom;ctx.clearRect(0,0,c.width,c.height);for(let i=0;i<b.points_xyz_m.length;i++){const p=b.points_xyz_m[i];if(Math.hypot(...p)>10||b.point_history_slots[i]>Number($("history").value))continue;let x=p[0],y=kind==="xz"?p[2]:p[1];if(kind==="three"){x=Math.cos(yaw)*p[0]-Math.sin(yaw)*p[1];y=Math.cos(pitch)*p[2]-Math.sin(pitch)*(Math.sin(yaw)*p[0]+Math.cos(yaw)*p[1]);}ctx.fillStyle=colors[b.point_history_slots[i]];ctx.fillRect(c.width/2+x*scale,c.height/2-y*scale,2,2);}ctx.fillStyle="#fff";ctx.fillText("传感器原点 +（米）",c.width/2,c.height/2);}}
  $("bundle").onchange=guard(async e=>{if(session)throw Error("已有观察，请下载记录后刷新");const [b,h]=await read(e.target.files[0]);session=new SurfaceBrowserReview(b,h,$("reviewer").value);$("bundle").disabled=true;$("reviewer").disabled=true;set(emptyTarget(b));render();status("单个五帧观察已载入，尚未完成人工判断。");});
  $("anchor").onclick=guard(()=>{if(!session||session.blind)throw Error("没有可编辑观察");const a=edit();a.anchors.push({position_m:[null,null,null],evidence:""});a.membership.forEach(r=>r.push(null));set(a);});
  $("opening").onclick=guard(()=>{if(!session||session.blind)throw Error("没有可编辑观察");const a=edit();a.openings.push({position_m:[null,null,null],direction:null,width_m:null,height_m:null,evidence:""});a.membership.push(a.anchors.map(()=>null));set(a);});
  $("commit").onclick=guard(()=>{if(!session)throw Error("先载入观察");session.commit(edit());$("annotation").disabled=true;$("commit").disabled=true;$("anchor").disabled=true;$("opening").disabled=true;$("reference").disabled=false;status("盲看已锁定，可下载并核对独立参考。");});
  $("reference").onchange=guard(async e=>{const [r,h]=await read(e.target.files[0]);session.reveal(r,h);$("referenceView").textContent=JSON.stringify(r.construction_reference,null,2);$("reference").disabled=true;$("notes").disabled=false;$("finish").disabled=false;});
  $("finish").onclick=guard(()=>{session.finish($("notes").value);$("notes").disabled=true;$("finish").disabled=true;status("核对已记录；不自动获得训练资格。");});
  $("save").onclick=guard(()=>{if(!session)throw Error("先载入观察");const u=URL.createObjectURL(new Blob([JSON.stringify(session.export(),null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=u;a.download="surface-review.json";a.click();URL.revokeObjectURL(u);});
  $("history").oninput=render;$("three").onpointerdown=e=>{drag=[e.clientX,e.clientY];$("three").setPointerCapture(e.pointerId);};$("three").onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];render();};$("three").onpointerup=()=>drag=null;$("three").onwheel=e=>{e.preventDefault();zoom=Math.max(.2,Math.min(10,zoom*Math.exp(-e.deltaY*.001)));render();};
}
