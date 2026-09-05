#!/usr/bin/env python3
"""Phase-3 M1 train-only ray-column-dropout corrective trainer."""

from __future__ import annotations

import argparse,json,math,random,time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.phase3_multitask_dataset import CanoV2RMultitaskDataset
from mtare_topo.evaluation.phase3_semantic_metrics import confusion_matrix,decode_direction_components,match_headings,per_class_scores,same_cluster_cosine
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet,multitask_loss
from mtare_topo.representation.ray_column_dropout import apply_ray_column_dropout


def collate(samples):
    return {
        "student":torch.from_numpy(np.stack([x["student"] for x in samples])),
        "direction_target":torch.from_numpy(np.stack([x["direction_target"] for x in samples])),
        "count_target":torch.tensor([x["count_target"] for x in samples],dtype=torch.long),
        "role_target":torch.tensor([x["role_target"] for x in samples],dtype=torch.long),
        "frame_id":[x["frame_id"] for x in samples],"cluster_id":[x["cluster_id"] for x in samples],"parent_id":[x["parent_id"] for x in samples],
        "headings_robot_deg":[x["headings_robot_deg"] for x in samples],
    }


def seed_everything(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True,warn_only=True)


class BlockShuffleSampler(Sampler[int]):
    """Shuffle contiguous blocks while visiting every frame exactly once."""
    def __init__(self,size:int,block_size:int,seed:int):self.size=size;self.block_size=block_size;self.seed=seed;self.epoch=0
    def __len__(self):return self.size
    def set_epoch(self,epoch:int):self.epoch=epoch
    def __iter__(self):
        blocks=list(range(0,self.size,self.block_size));rng=random.Random(self.seed+self.epoch);rng.shuffle(blocks)
        for start in blocks:
            indices=list(range(start,min(start+self.block_size,self.size)));rng.shuffle(indices);yield from indices


@torch.no_grad()
def evaluate(model,loader,device,role_weights,preserve_arrays=False):
    model.eval();loss_sums={k:0.0 for k in ("total","direction","count","role")};n=0;roles=[];role_pred=[];counts=[];count_pred=[];emb=[];clusters=[];direction_bce=[];matched=predicted=truth=0;angular=[]
    all_direction=[];all_direction_target=[];all_role_logits=[];all_count_logits=[];frame_ids=[];parent_ids=[]
    for batch in loader:
        student=batch["student"].to(device,non_blocking=True);direction=batch["direction_target"].to(device);count=batch["count_target"].to(device);role=batch["role_target"].to(device);outputs=model(student);losses=multitask_loss(outputs,direction,count,role,role_weights)
        size=len(student);n+=size
        for key,value in losses.items():loss_sums[key]+=float(value)*size
        roles.extend(role.cpu().tolist());role_pred.extend(outputs["role_logits"].argmax(1).cpu().tolist());counts.extend(count.cpu().tolist());count_pred.extend(outputs["count_logits"].argmax(1).cpu().tolist());emb.append(outputs["z_role"].cpu().numpy());clusters.extend(batch["cluster_id"])
        direction_bce.extend(torch.nn.functional.binary_cross_entropy_with_logits(outputs["direction_logits"],direction,reduction="none").mean(1).cpu().tolist())
        for logits,headings in zip(outputs["direction_logits"].cpu().numpy(),batch["headings_robot_deg"]):
            m,p,t,e=match_headings(decode_direction_components(logits,.5),headings);matched+=m;predicted+=p;truth+=t;angular.extend(e)
        if preserve_arrays:
            all_direction.append(outputs["direction_logits"].cpu().numpy().astype(np.float16));all_direction_target.append(direction.cpu().numpy().astype(np.float16));all_role_logits.append(outputs["role_logits"].cpu().numpy().astype(np.float16));all_count_logits.append(outputs["count_logits"].cpu().numpy().astype(np.float16));frame_ids.extend(batch["frame_id"]);parent_ids.extend(batch["parent_id"])
    role_matrix=confusion_matrix(np.asarray(roles),np.asarray(role_pred),3);count_matrix=confusion_matrix(np.asarray(counts),np.asarray(count_pred),6)
    precision=matched/max(predicted,1);recall=matched/max(truth,1);f1=2*precision*recall/max(precision+recall,1e-12)
    count_scores=per_class_scores(count_matrix);formal=np.asarray(count_scores["f1"][:4]);present=np.asarray(count_scores["support"][:4])>0;count_scores["macro_f1_count_1_to_4"]=float(formal[present].mean())
    result={"loss":{k:value/max(n,1) for k,value in loss_sums.items()},"role":{"confusion_matrix":role_matrix.tolist(),**per_class_scores(role_matrix)},"count":{"confusion_matrix":count_matrix.tolist(),**count_scores},"direction":{"mean_bce":float(np.mean(direction_bce)),"threshold":.5,"matching_tolerance_deg":20.0,"matched":matched,"predicted":predicted,"truth":truth,"precision":precision,"recall":recall,"f1":f1,"mean_matched_angular_error_deg":float(np.mean(angular)) if angular else None},"representation":{"same_cluster_cosine":same_cluster_cosine(np.concatenate(emb),clusters)},"frames":n}
    if preserve_arrays:
        result["arrays"]={"direction_logits":np.concatenate(all_direction),"direction_target":np.concatenate(all_direction_target),"role_logits":np.concatenate(all_role_logits),"count_logits":np.concatenate(all_count_logits),"role_target":np.asarray(roles,dtype=np.int8),"count_target":np.asarray(counts,dtype=np.int8),"embedding":np.concatenate(emb).astype(np.float16),"frame_id":np.asarray(frame_ids,dtype='U128'),"cluster_id":np.asarray(clusters,dtype='U128'),"parent_id":np.asarray(parent_ids,dtype='U128')}
    return result


@torch.no_grad()
def stability_audit(model,loader,device,maximum_frames=2048):
    model.eval();normal=[];masked=[];rotation_direction_max=0.0;rotation_z_min=1.0;seen=0
    for batch in loader:
        student=batch["student"].to(device);base=model(student);perturbed=student.clone();perturbed[:,:1,:,::10]=1.0;perturbed[:,1:,:,::10]=0.0;changed=model(perturbed)
        normal.append(base["z_role"].cpu());masked.append(changed["z_role"].cpu())
        rotated=model(torch.roll(student,12,dims=-1));rotation_direction_max=max(rotation_direction_max,float((torch.roll(base["direction_logits"],12,dims=-1)-rotated["direction_logits"]).abs().max().cpu()));rotation_z_min=min(rotation_z_min,float(torch.nn.functional.cosine_similarity(base["z_role"],rotated["z_role"]).min().cpu()));seen+=len(student)
        if seen>=maximum_frames:break
    normal=torch.cat(normal)[:maximum_frames];masked=torch.cat(masked)[:maximum_frames];cos=torch.nn.functional.cosine_similarity(normal,masked).numpy()
    return {"frames":len(cos),"fixed_masking_z_role_cosine_mean":float(cos.mean()),"fixed_masking_z_role_cosine_p05":float(np.quantile(cos,.05)),"rotation_shift_deg":6.0,"rotation_direction_max_absolute_logit_error":rotation_direction_max,"rotation_z_role_minimum_cosine":rotation_z_min}


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset-run",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--mode",choices=("M1D",),required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--epochs",type=int,default=30);p.add_argument("--batch-size",type=int,default=128);p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--patience",type=int,default=6);p.add_argument("--workers",type=int,default=0);a=p.parse_args()
    if a.epochs<1 or a.seed not in (0,1,2):raise ValueError("formal contract requires seed 0/1/2 and positive epochs")
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);seed_everything(a.seed);device=torch.device("cuda")
    train=CanoV2RMultitaskDataset(a.dataset_run,"train");validation=CanoV2RMultitaskDataset(a.dataset_run,"validation")
    sampler=BlockShuffleSampler(len(train),1024,a.seed);train_loader=DataLoader(train,batch_size=a.batch_size,sampler=sampler,num_workers=a.workers,collate_fn=collate,pin_memory=True);validation_loader=DataLoader(validation,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,collate_fn=collate,pin_memory=True)
    role_index={"interior":0,"junction":1,"terminal":2};role_counts=np.bincount([role_index[str(item["primary_role"])] for item in train.records],minlength=3);role_weights=torch.tensor(role_counts.sum()/(3*np.maximum(role_counts,1)),dtype=torch.float32,device=device)
    model=StructuralSemanticNet().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay);history=[];best=-math.inf;stale=0;started=time.monotonic();augmentation_generator=torch.Generator().manual_seed(100000+a.seed);augmentation_totals={"samples":0,"augmented_samples":0,"phase_counts":[0]*10}
    for epoch in range(1,a.epochs+1):
        sampler.set_epoch(epoch)
        model.train();sums={k:0.0 for k in ("total","direction","count","role")};seen=0
        for batch in train_loader:
            student,augmentation=apply_ray_column_dropout(batch["student"],augmentation_generator,probability=.5,period=10);student=student.to(device,non_blocking=True);direction=batch["direction_target"].to(device);count=batch["count_target"].to(device);role=batch["role_target"].to(device);optimizer.zero_grad(set_to_none=True);outputs=model(student);losses=multitask_loss(outputs,direction,count,role,role_weights)
            objective=losses["total"];objective.backward();optimizer.step();size=len(student);seen+=size;augmentation_totals["samples"]+=augmentation["samples"];augmentation_totals["augmented_samples"]+=augmentation["augmented_samples"];augmentation_totals["phase_counts"]=[a+b for a,b in zip(augmentation_totals["phase_counts"],augmentation["phase_counts"])]
            for key,value in losses.items():sums[key]+=float(value.detach())*size
        metrics=evaluate(model,validation_loader,device,role_weights);record={"epoch":epoch,"train_loss":{k:v/seen for k,v in sums.items()},"validation":metrics};history.append(record);(out/"epoch_metrics.jsonl").open("a").write(json.dumps(record,separators=(",",":"))+"\n")
        score=metrics["direction"]["f1"]
        if score>best+1e-8:best=score;stale=0;torch.save({"model":model.state_dict(),"epoch":epoch,"seed":a.seed,"mode":a.mode,"config":vars(a)},out/"best.pt")
        else:stale+=1
        if stale>=a.patience:break
    torch.save({"model":model.state_dict(),"epoch":history[-1]["epoch"],"seed":a.seed,"mode":a.mode,"config":vars(a)},out/"last.pt")
    checkpoint=torch.load(out/"best.pt",map_location=device,weights_only=False);model.load_state_dict(checkpoint["model"]);final=evaluate(model,validation_loader,device,role_weights,preserve_arrays=True);arrays=final.pop("arrays");np.savez_compressed(out/"validation_outputs.npz",**arrays);stability_cuda=stability_audit(model,validation_loader,device);model=model.to("cpu");stability_cpu=stability_audit(model,validation_loader,torch.device("cpu"));stability={"cuda":stability_cuda,"cpu":stability_cpu};(out/"best_validation_metrics.json").write_text(json.dumps(final,indent=2)+"\n");(out/"representation_stability.json").write_text(json.dumps(stability,indent=2)+"\n")
    summary={"status":"COMPLETED_PHASE3_MASKING_CORRECTIVE_EXECUTOR","mode":a.mode,"seed":a.seed,"epochs_completed":len(history),"best_epoch":int(checkpoint["epoch"]),"best_direction_validation_f1":best,"best_validation":final,"stability":stability,"augmentation":{"scope":"train_only","probability":.5,"period_columns":10,"masked_range_value_normalized":1.0,"masked_valid_value":0.0,**augmentation_totals},"parameters":sum(p.numel() for p in model.parameters()),"duration_seconds":time.monotonic()-started,"train_frames":len(train),"validation_frames":len(validation),"strict_test_frames_read":0,"mtare_frames_read":0}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str)+"\n");print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
