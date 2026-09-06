"use strict";
// No network, no bundled teacher, and no automatic annotation. Server-side
// (Python) import remains authoritative for semantic and source validation.
const emptyAnnotation = () => ({structures:[], complete_regions:[], unknown_regions:[], notes:""});
function validateAnnotation(a) {
  const keys=(o,n)=>{if(!o||Array.isArray(o)||Object.keys(o).sort().join()!==n.sort().join())throw Error("标注字段不完整或含额外字段");};
  const xyz=p=>{if(!Array.isArray(p)||p.length!==3||p.some(x=>typeof x!=="number"||!Number.isFinite(x)))throw Error("请填写三个有限坐标");};
  const text=t=>{if(typeof t!=="string"||!t.trim()||t.includes("请填写"))throw Error("请填写具体可见依据");};
  keys(a,["structures","complete_regions","unknown_regions","notes"]);
  if(typeof a.notes!=="string"||!Array.isArray(a.structures)||a.structures.length>32)throw Error("结构数量或备注非法");
  let count=0;const ids=new Set();
  for(const s of a.structures){keys(s,["local_id","event","center_xyz_m","openings","evidence_note"]);if(!Number.isSafeInteger(s.local_id)||s.local_id<0||ids.has(s.local_id))throw Error("结构编号重复或非法");ids.add(s.local_id);if(!["corridor","junction","terminal","unknown"].includes(s.event))throw Error("结构类型非法");if(s.center_xyz_m!==null)xyz(s.center_xyz_m);text(s.evidence_note);if(!Array.isArray(s.openings))throw Error("开口必须为列表");count+=s.openings.length;for(const p of s.openings){keys(p,["position_xyz_m","direction","width_m","height_m","evidence_note"]);if(p.position_xyz_m!==null)xyz(p.position_xyz_m);if(p.direction!==null){xyz(p.direction);if(Math.abs(Math.hypot(...p.direction)-1)>1e-5)throw Error("方向必须为单位向量或null");}for(const d of [p.width_m,p.height_m])if(d!==null&&(typeof d!=="number"||!Number.isFinite(d)||d<=0))throw Error("宽高只能为正数或null");text(p.evidence_note);}}
  if(count>64)throw Error("超过64开口，不能截断标注");
  for(const name of ["complete_regions","unknown_regions"]){if(!Array.isArray(a[name]))throw Error("区域必须为列表");for(const r of a[name]){keys(r,["min_xyz_m","max_xyz_m","reason"]);xyz(r.min_xyz_m);xyz(r.max_xyz_m);text(r.reason);if(r.min_xyz_m.some((v,i)=>v>=r.max_xyz_m[i]))throw Error("区域范围必须为正");}}
  for(const c of a.complete_regions)for(const u of a.unknown_regions)if([0,1,2].every(i=>Math.max(c.min_xyz_m[i],u.min_xyz_m[i])<Math.min(c.max_xyz_m[i],u.max_xyz_m[i])))throw Error("完整区域与未知区域重叠");
  return a;
}
function validateBundle(b) {
  const keys = (o, expected) => {
    if (!o || Array.isArray(o) || Object.keys(o).sort().join() !== expected.sort().join()) throw Error("字段不符合盲看合同，不能混入教师或地图信息");
  };
  keys(b,["schema","bundle_id","coordinate_frame","decisions"]);
  if (b.schema !== "gse_structure_blind_bundle_v1" || b.coordinate_frame !== "current_sensor_m" || typeof b.bundle_id !== "string" || !b.bundle_id.trim() || !Array.isArray(b.decisions) || b.decisions.length !== 21) throw Error("需要21个决策位置的有效盲看包");
  let last=null,lastSource=null; const keyOrder=new Map(),orderKey=new Map();
  for(const d of b.decisions) {
    keys(d,["decision_index","source_frame_keys","source_order_indices","points_xyz_m"]);
    if(!Number.isSafeInteger(d.decision_index)||d.decision_index<0||(last!==null&&d.decision_index!==last+1))throw Error("决策位置必须连续"); last=d.decision_index;
    const f=d.source_frame_keys,o=d.source_order_indices;
    if(!Array.isArray(f)||!Array.isArray(o)||f.length!==5||o.length!==5||new Set(f).size!==5||f.some(x=>typeof x!=="string"||!x.trim())||o.some((x,i)=>!Number.isSafeInteger(x)||x<0||(i>0&&x<=o[i-1])))throw Error("五帧身份或因果顺序错误");
    if(lastSource!==null&&o[4]<=lastSource)throw Error("整段源时间必须向前"); lastSource=o[4];
    f.forEach((k,i)=>{if((keyOrder.has(k)&&keyOrder.get(k)!==o[i])||(orderKey.has(o[i])&&orderKey.get(o[i])!==k))throw Error("源帧身份发生漂移");keyOrder.set(k,o[i]);orderKey.set(o[i],k);});
    if(!Array.isArray(d.points_xyz_m)||d.points_xyz_m.length>57600||d.points_xyz_m.some(p=>!Array.isArray(p)||p.length!==3||p.some(x=>typeof x!=="number"||!Number.isFinite(x))))throw Error("点云数量或坐标非法");
  }
  return b;
}
class BrowserReview {
  constructor(bundle, fileSha, reviewer) {this.bundle=validateBundle(JSON.parse(JSON.stringify(bundle)));if(typeof reviewer!=="string"||!reviewer.trim())throw Error("请填写复核人");this.fileSha=fileSha;this.reviewer=reviewer;this.records=[];this.reference=null;this.referenceSha=null;this.notes=null;}
  current(){return this.records.length<21?JSON.parse(JSON.stringify(this.bundle.decisions[this.records.length])):null;}
  commit(index,annotation){if(!this.current()||this.current().decision_index!==index||this.reference)throw Error("盲看提交顺序错误或已锁定");validateAnnotation(annotation);this.records.push({decision_index:index,annotation:JSON.parse(JSON.stringify(annotation))});}
  reveal(reference,sha){if(this.records.length!==21||this.reference)throw Error("必须先完成全部盲看，参考只加载一次");if(reference.schema!=="gse_structure_reference_v1"||reference.bundle_id!==this.bundle.bundle_id||!Array.isArray(reference.decisions)||reference.decisions.length!==21||reference.decisions.some((d,i)=>d.decision_index!==this.bundle.decisions[i].decision_index))throw Error("参考与盲看包不对应");this.reference=JSON.parse(JSON.stringify(reference));this.referenceSha=sha;}
  finish(notes){if(!this.reference||this.notes!==null||typeof notes!=="string"||!notes.trim())throw Error("需要一次有效的参考核对说明");this.notes=notes;}
  export(){return JSON.parse(JSON.stringify({schema:"gse_structure_browser_review_v1",bundle_id:this.bundle.bundle_id,blind_bundle_file_sha256:this.fileSha,reviewer_assertion:this.reviewer,blind_records:this.records,reference_file_sha256:this.referenceSha,reference_notes:this.notes,automatic_training_eligibility:false}));}
}
if(typeof module!=="undefined"&&module.exports)module.exports={BrowserReview,validateBundle,validateAnnotation,emptyAnnotation};
if(typeof document!=="undefined") {
  const $=id=>document.getElementById(id); let session=null,view=0,yaw=.3,pitch=.45,zoom=1,drag=null,moved=false;
  const status=t=>{$("status").textContent=t;};
  const annotation=()=>JSON.parse($("annotation").value);
  const setAnnotation=a=>{$("annotation").value=JSON.stringify(a,null,2);};
  const sha=async buffer=>Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",buffer)),b=>b.toString(16).padStart(2,"0")).join("");
  const read=async file=>{if(!file||file.size>128*1024*1024)throw Error("文件不存在或超过128MiB，先按片段导出");const bytes=await file.arrayBuffer();return [JSON.parse(new TextDecoder().decode(bytes)),await sha(bytes)];};
  const guard=fn=>async e=>{try{await fn(e);}catch(error){status("未完成："+error.message);}};
  function project(p,kind,w,h,scale) {let a,b;if(kind==="xy"){[a,b]=p;}else if(kind==="xz"){a=p[0];b=p[2];}else{a=Math.cos(yaw)*p[0]-Math.sin(yaw)*p[1];const y=Math.sin(yaw)*p[0]+Math.cos(yaw)*p[1];b=Math.cos(pitch)*p[2]-Math.sin(pitch)*y;}return[w/2+a*scale,h/2-b*scale];}
  function render(){
    if(!session)return;const d=session.bundle.decisions[view];$("step").textContent=`观测 ${view+1}/21 · ${session.reference?"参考阶段（盲标已锁定）":"盲看阶段"}`;
    for(const kind of ["three","xy","xz"]){const c=$(kind),ctx=c.getContext("2d"),extent=Math.max(5,...d.points_xyz_m.map(p=>Math.max(Math.abs(p[0]),Math.abs(p[1]),Math.abs(p[2]))));const scale=Math.min(c.width,c.height)*.45/extent*zoom;ctx.clearRect(0,0,c.width,c.height);ctx.strokeStyle="#43596c";ctx.beginPath();ctx.moveTo(0,c.height/2);ctx.lineTo(c.width,c.height/2);ctx.moveTo(c.width/2,0);ctx.lineTo(c.width/2,c.height);ctx.stroke();ctx.fillStyle="#7bd7ed";for(const p of d.points_xyz_m){const [x,y]=project(p,kind,c.width,c.height,scale);ctx.fillRect(x,y,2,2);}ctx.fillStyle="#ffb666";ctx.fillRect(c.width/2-3,c.height/2-3,6,6);c._projection={scale,points:d.points_xyz_m};}
    if(session.reference)$("referenceView").textContent=JSON.stringify({blind:session.records[view].annotation,reference:session.reference.decisions[view].reference_annotation},null,2);
  }
  $("bundle").onchange=guard(async e=>{if(session)throw Error("当前片段已载入；先下载记录，再刷新开始另一片段");const [b,h]=await read(e.target.files[0]);session=new BrowserReview(b,h,$("reviewer").value);$("reviewer").disabled=true;$("bundle").disabled=true;setAnnotation(emptyAnnotation());render();status("已载入匿名盲看包。只提交当前观测，不查看未来帧或构造参考。");});
  $("addStructure").onclick=guard(()=>{const a=annotation();a.structures.push({local_id:a.structures.length,event:$("event").value,center_xyz_m:null,openings:[],evidence_note:"请填写可见依据"});setAnnotation(a);});
  $("addOpening").onclick=guard(()=>{const a=annotation();if(!a.structures.length)throw Error("先添加所属结构");a.structures.at(-1).openings.push({position_xyz_m:null,direction:null,width_m:null,height_m:null,evidence_note:"请填写开口依据"});setAnnotation(a);});
  for(const [button,key] of [["addUnknown","unknown_regions"],["addComplete","complete_regions"]])$(button).onclick=guard(()=>{const a=annotation();a[key].push({min_xyz_m:[null,null,null],max_xyz_m:[null,null,null],reason:"请填写实际范围及依据"});setAnnotation(a);});
  $("commit").onclick=guard(()=>{if(!session||!session.current())throw Error("没有可提交的盲看观测");const a=annotation();if(!Array.isArray(a.structures)||!Array.isArray(a.complete_regions)||!Array.isArray(a.unknown_regions)||typeof a.notes!=="string")throw Error("判断字段不完整");if(JSON.stringify(a).includes("请填写"))throw Error("请补充依据或删除模板，不提交占位文字");session.commit(session.current().decision_index,a);if(session.current()){view=session.records.length;setAnnotation(emptyAnnotation());render();status("上一帧已锁定，请独立判断下一帧。");}else{$("commit").disabled=true;$("annotation").disabled=true;for(const id of ["addStructure","addOpening","addUnknown","addComplete"])$(id).disabled=true;$("reference").disabled=false;status("21帧盲看已完成并锁定。请先下载记录，再加载独立构造参考。");}});
  $("reference").onchange=guard(async e=>{const [r,h]=await read(e.target.files[0]);session.reveal(r,h);$("reference").disabled=true;$("referenceNotes").disabled=false;$("finish").disabled=false;$("previous").disabled=false;$("next").disabled=false;view=0;render();status("构造参考已显示。不得回改盲看判断；几何修正建议写入单独核对说明。");});
  $("previous").onclick=()=>{view=Math.max(0,view-1);render();};$("next").onclick=()=>{view=Math.min(20,view+1);render();};
  $("finish").onclick=guard(()=>{session.finish($("referenceNotes").value);$("referenceNotes").disabled=true;$("finish").disabled=true;status("核对说明已锁定。下载记录后仍需Python校验，不自动获得训练资格。");});
  $("save").onclick=guard(()=>{if(!session)throw Error("没有复核记录");const url=URL.createObjectURL(new Blob([JSON.stringify(session.export(),null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="gse-review-record.json";a.click();URL.revokeObjectURL(url);});
  $("three").onpointerdown=e=>{drag=[e.clientX,e.clientY];moved=false;$("three").setPointerCapture(e.pointerId);};
  $("three").onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag[0],dy=e.clientY-drag[1];moved=moved||Math.abs(dx)+Math.abs(dy)>2;yaw+=dx*.008;pitch+=dy*.008;drag=[e.clientX,e.clientY];render();};
  $("three").onpointerup=()=>{drag=null;};$("three").onwheel=e=>{e.preventDefault();zoom=Math.min(5,Math.max(.2,zoom*Math.exp(-e.deltaY*.001)));render();};
  for(const kind of ["three","xy","xz"])$(kind).onclick=e=>{const c=$(kind);if(!c._projection||moved&&kind==="three")return;const box=c.getBoundingClientRect(),x=(e.clientX-box.left)*c.width/box.width,y=(e.clientY-box.top)*c.height/box.height;let best=null,dist=144;for(const p of c._projection.points){const [u,v]=project(p,kind,c.width,c.height,c._projection.scale),d=(x-u)**2+(y-v)**2;if(d<dist){dist=d;best=p;}}if(best)$("picked").textContent="所选表面点 XYZ（米）："+best.map(x=>x.toFixed(3)).join(", ")+"；这不是自动生成的中心/开口标签。";};
  setAnnotation(emptyAnnotation());
}
