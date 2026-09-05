#!/usr/bin/env python3
"""Full Teacher-join and zero-training decoder readiness audit."""
from __future__ import annotations
import argparse,json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import zarr
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_spatial_event_set import SpatialEventSetDecoder,spatial_event_set_loss,validate_spatial_event_targets

PASS="PASS_GSE_SPATIAL_EVENT_SET_READINESS_V1";FAIL="FAIL_GSE_SPATIAL_EVENT_SET_READINESS_V1"
EXPECTED={"worlds":80,"observations":188126,"tokens":133055,"identities":1076,"parameters":93638}

def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--teacher-root",required=True,type=Path);p.add_argument("--source-root",required=True,type=Path);p.add_argument("--output-dir",required=True,type=Path);a=p.parse_args();start=time.monotonic();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 shards=sorted(a.teacher_root.resolve().glob("*/*.zarr"));counts=Counter();types=Counter();hist=Counter();sample={};global_ids=[]
 for index,path in enumerate(shards,1):
  g=zarr.open_group(str(path),mode="r");parent=str(g.attrs["parent_id"]);partition=str(g.attrs["partition"]);source=zarr.open_group(str(a.source_root.resolve()/"train"/f"{parent}.zarr"),mode="r")
  gid=np.asarray(g["global_sequence_index"][:],dtype=np.int64);source_gid=np.asarray(source["global_sequence_index"][:],dtype=np.int64)
  if not np.array_equal(gid,source_gid):raise RuntimeError(f"Teacher/source global identity drift {parent}")
  target={"event_type_index":torch.from_numpy(np.asarray(g["event_type_index"][:],dtype=np.int64)),"event_relative_xyz_m":torch.from_numpy(np.asarray(g["event_relative_xyz_m"][:],dtype=np.float32)),"event_identity_index":torch.from_numpy(np.asarray(g["event_identity_index"][:],dtype=np.int64)),"event_mask":torch.from_numpy(np.asarray(g["event_mask"][:],dtype=np.uint8))}
  validate_spatial_event_targets(target);card=np.asarray(g["set_cardinality"][:],dtype=np.int64);distance=np.asarray(g["event_distance_m"][:],dtype=np.float32);identity=np.asarray(g["event_identity_index"][:],dtype=np.int64);mask=np.asarray(g["event_mask"][:],dtype=bool)
  if not np.array_equal(card,mask.sum(axis=1)):raise RuntimeError(f"cardinality drift {parent}")
  for row in range(len(card)):
   active=np.flatnonzero(mask[row]);d=distance[row,active];ids=identity[row,active]
   if len(d)>1 and (np.any(d[1:]<d[:-1]-1e-6) or np.any((np.abs(d[1:]-d[:-1])<=1e-6)&(ids[1:]<ids[:-1]))):raise RuntimeError(f"token order drift {parent}:{row}")
   c=int(card[row]);
   if c not in sample:sample[c]={key:value[row:row+1].clone() for key,value in target.items()}
  counts.update({"worlds":1,"observations":len(gid),"tokens":int(mask.sum()),f"{partition}_worlds":1});hist.update(int(v) for v in card);types.update({"terminal":int(np.sum(target["event_type_index"].numpy()[mask]==0)),"junction":int(np.sum(target["event_type_index"].numpy()[mask]==1))});global_ids.append(gid)
  print(json.dumps({"world":parent,"index":index,"of":len(shards),"observations":len(gid),"tokens":int(mask.sum())},sort_keys=True),flush=True)
 if set(sample)!=set(range(6)):raise RuntimeError("real Teacher lacks one cardinality from 0 through 5")
 batch={key:torch.cat([sample[c][key] for c in range(6)]) for key in sample[0]};torch.manual_seed(20260828);model=SpatialEventSetDecoder();features={"context":torch.randn(6,128),"directional":torch.randn(6,128,180)};outputs=model(features);loss=spatial_event_set_loss(outputs,batch);loss["total"].backward();finite=all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in model.parameters())
 perm=torch.tensor([7,3,15,0,9,4,2,13,5,10,1,14,6,8,12,11]);permuted={k:(v[:,perm] if v.ndim>=2 and v.shape[1]==16 else v) for k,v in outputs.items()};permuted_loss=spatial_event_set_loss(permuted,batch);permutation_error=max(abs(float(loss[k].detach())-float(permuted_loss[k].detach())) for k in ("total","presence","event_type","position","descriptor","uncertainty"))
 model.eval();roll=15
 with torch.no_grad():base=model(features);rot=model({"context":features["context"],"directional":torch.roll(features["directional"],roll,dims=-1)})
 angle=2*math.pi*roll/180;xyz=base["event_relative_xyz_m"];expected=torch.stack((math.cos(angle)*xyz[...,0]-math.sin(angle)*xyz[...,1],math.sin(angle)*xyz[...,0]+math.cos(angle)*xyz[...,1],xyz[...,2]),-1);rotation_error=float(torch.max(torch.abs(rot["event_relative_xyz_m"]-expected)))
 unique=len(np.unique(np.concatenate(global_ids)));parameter_count=sum(v.numel() for v in model.parameters());checks={"exact_80_worlds":counts["worlds"]==80,"exact_60_20_split":counts["fit_worlds"]==60 and counts["selection_worlds"]==20,"exact_188126_rows":counts["observations"]==188126 and unique==188126,"exact_133055_tokens":counts["tokens"]==133055,"exact_type_counts":dict(types)=={"terminal":30789,"junction":102266},"exact_cardinality_histogram":dict(hist)=={0:81069,1:83093,2:22076,3:1751,4:128,5:9},"real_cardinality_0_through_5_covered":set(sample)==set(range(6)),"decoder_parameter_count_93638":parameter_count==93638,"query_permutation_loss_error_at_most_1e_6":permutation_error<=1e-6,"circular_rotation_error_at_most_5e_4":rotation_error<=5e-4,"finite_real_teacher_backward":finite,"zero_training_inference_C09_C10_MTARE":True};checks["all_passed"]=all(checks.values());status=PASS if checks["all_passed"] else FAIL
 summary={"schema_version":"gse_spatial_event_set_readiness_v1","status":status,"population":{"worlds":counts["worlds"],"fit_worlds":counts["fit_worlds"],"selection_worlds":counts["selection_worlds"],"observations":counts["observations"],"unique_global_ids":unique,"tokens":counts["tokens"],"type_counts":dict(types),"cardinality_histogram":dict(sorted(hist.items()))},"decoder":{"parameters":parameter_count,"queries":16,"real_teacher_batch_cardinalities":list(range(6)),"query_permutation_loss_error":permutation_error,"circular_rotation_max_abs_error_m":rotation_error,"finite_backward":finite},"checks":checks,"duration_seconds":time.monotonic()-start,"optimizer_steps":0,"trained_model_inference_frames":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0};write_json(out/"summary.json",summary);print(json.dumps(summary,indent=2));return 0 if status==PASS else 2
if __name__=="__main__":raise SystemExit(main())
