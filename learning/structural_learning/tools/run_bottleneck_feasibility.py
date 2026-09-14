from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from learning.structural_learning.bottleneck_model import ForcedGlobalBottleneckNet
from learning.structural_learning.dataset import StructuralSurfaceDataset
from learning.structural_learning.feasibility_model import completion_loss


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def seed_all(seed: int, deterministic: bool) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic: torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def collate(samples):
    return {"input_surface":torch.from_numpy(np.stack([s["input_surface"] for s in samples])),"teacher_surface":torch.from_numpy(np.stack([s["teacher_surface"] for s in samples]))}


def select(x,channels): return x[:,channels]


def degrade(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    out=x.clone(); batch,_,height,width=out.shape; rows,cols=torch.meshgrid(torch.arange(height,device=x.device),torch.arange(width,device=x.device),indexing="ij"); cy=(height-1)/2; cx=(width-1)/2
    for i in range(batch):
        mode=i%4
        if mode==0:
            keep=torch.rand((height,width),generator=generator,device=x.device)>0.40
        elif mode==1:
            angle=torch.atan2(cy-rows,cols-cx); center=(torch.rand((),generator=generator,device=x.device)*2-1)*math.pi; delta=torch.atan2(torch.sin(angle-center),torch.cos(angle-center)); keep=torch.abs(delta)>math.radians(35)
        elif mode==2:
            radius=torch.sqrt((rows-cy)**2+(cols-cx)**2); limit=torch.randint(30,46,(1,),generator=generator,device=x.device); keep=radius<=limit
        else:
            probability=torch.linspace(0.35,0.9,width,device=x.device)[None,:].expand(height,width); keep=torch.rand((height,width),generator=generator,device=x.device)<probability
        out[i]*=keep
    return out


def geometry_similarity_loss(latent: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    z=F.normalize(latent,dim=1); feature_similarity=(z@z.T+1.0)/2.0
    masks=teacher[:,0].flatten(1); intersection=masks@masks.T; sums=masks.sum(1); union=sums[:,None]+sums[None,:]-intersection; teacher_iou=intersection/(union+1.0)
    upper=torch.triu(torch.ones_like(feature_similarity,dtype=torch.bool),diagonal=1)
    return F.mse_loss(feature_similarity[upper],teacher_iou[upper])


def train_step(model,batch,optimizer,device,cfg,generator,with_constraints=True):
    model.train(); x=select(batch["input_surface"].to(device),cfg["model"]["input_channels"]); teacher=batch["teacher_surface"].to(device); optimizer.zero_grad(set_to_none=True)
    pred,z=model(x,return_features=True); base,_=completion_loss(pred,teacher,float(cfg["loss"]["positive_weight"])); total=base; parts={"completion":float(base.detach())}
    if with_constraints:
        x2=degrade(x,generator); pred2,z2=model(x2,return_features=True); deg,_=completion_loss(pred2,teacher,float(cfg["loss"]["positive_weight"])); consistency=(1-F.cosine_similarity(z,z2,dim=1)).mean(); geometry=geometry_similarity_loss(z,teacher)
        total=base+float(cfg["loss"]["degraded_completion_weight"])*deg+float(cfg["loss"]["latent_consistency_weight"])*consistency+float(cfg["loss"]["teacher_geometry_weight"])*geometry
        parts.update({"degraded_completion":float(deg.detach()),"consistency":float(consistency.detach()),"geometry":float(geometry.detach())})
    if not torch.isfinite(total): raise RuntimeError("non-finite loss")
    total.backward(); optimizer.step(); parts["total"]=float(total.detach()); return parts


def count(pred,teacher,threshold):
    p=pred[:,0]>=threshold; t=teacher[:,0]>=.5; return np.asarray([float((p&t).sum()),float((p&~t).sum()),float((~p&t).sum())])


def metrics(c):
    tp,fp,fn=c; return {"iou":float(tp/max(tp+fp+fn,1)),"dice":float(2*tp/max(2*tp+fp+fn,1)),"precision":float(tp/max(tp+fp,1)),"recall":float(tp/max(tp+fn,1))}


@torch.no_grad()
def evaluate(model,loader,device,cfg):
    model.eval(); losses=[]; mc=np.zeros(3); bc=np.zeros(3)
    for batch in loader:
        x4=batch["input_surface"].to(device); x=select(x4,cfg["model"]["input_channels"]); teacher=batch["teacher_surface"].to(device); pred=model(x); loss,_=completion_loss(pred,teacher,float(cfg["loss"]["positive_weight"])); losses.append(float(loss)); mc+=count(pred,teacher,float(cfg["audit"]["surface_threshold"])); bc+=count(x4,teacher,float(cfg["audit"]["surface_threshold"]))
    return {"loss":float(np.mean(losses)),"model":metrics(mc),"input_copy_baseline":metrics(bc)}


def rank(v):
    order=np.argsort(v,kind="mergesort"); r=np.empty_like(order,dtype=float); r[order]=np.arange(len(v)); return r


def spearman(a,b): return float(np.corrcoef(rank(a),rank(b))[0,1])


def describe(v):
    a=np.asarray(v,dtype=float); return {"count":int(len(a)),"mean":float(a.mean()),"median":float(np.median(a)),"p10":float(np.percentile(a,10)),"p90":float(np.percentile(a,90)),"max":float(a.max())}


@torch.no_grad()
def collect(model,dataset,device,channels):
    features=[]; cell=[]; input_masks=[]; teacher_masks=[]
    for start in range(0,len(dataset),32):
        samples=[dataset[i] for i in range(start,min(start+32,len(dataset)))]; x4_np=np.stack([s["input_surface"] for s in samples]); x=select(torch.from_numpy(x4_np).to(device),channels); _,z=model(x,return_features=True); features.append(F.normalize(z,dim=1).cpu().numpy()); masks=x4_np[:,0]>.5; cell.extend(masks.sum(axis=(1,2))); input_masks.extend([m.reshape(-1) for m in masks]); teacher_masks.extend([(s["teacher_surface"][0]>.5).reshape(-1) for s in samples])
    return np.concatenate(features),np.asarray(cell),np.stack(input_masks),np.stack(teacher_masks)


def pair_data(features,cell,input_masks,teacher_masks,left,right):
    fd=1-np.sum(features[left]*features[right],axis=1); cell_diff=np.abs(cell[left]-cell[right])/np.maximum(np.maximum(cell[left],cell[right]),1); input_inter=np.logical_and(input_masks[left],input_masks[right]).sum(1); input_union=np.logical_or(input_masks[left],input_masks[right]).sum(1); coverage_shape=1-input_inter/np.maximum(input_union,1); ti=np.logical_and(teacher_masks[left],teacher_masks[right]).sum(1); tu=np.logical_or(teacher_masks[left],teacher_masks[right]).sum(1); geometry=1-ti/np.maximum(tu,1); return fd,cell_diff,coverage_shape,geometry


def make_perturbations(x):
    batch,_,h,w=x.shape; rows,cols=torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing="ij"); cy=(h-1)/2; cx=(w-1)/2; variants={}
    g=torch.Generator(device=x.device).manual_seed(20260808); random_drop=x.clone(); random_drop*=torch.rand((batch,1,h,w),generator=g,device=x.device)>.4; variants["random_surface_drop"]=random_drop
    angle=torch.atan2(cy-rows,cols-cx); sector=x.clone(); sector*=~((angle>-.6)&(angle<.6))[None,None]; variants["sector_occlusion"]=sector
    radius=torch.sqrt((rows-cy)**2+(cols-cx)**2); shortened=x.clone(); shortened*=radius.le(35)[None,None]; variants["shortened_visible_range"]=shortened
    probability=torch.linspace(.25,.9,w,device=x.device)[None,None,None,:]; sparse=x.clone(); sparse*=torch.rand((batch,1,h,w),generator=g,device=x.device)<probability; variants["nonuniform_sparsification"]=sparse
    return variants


@torch.no_grad()
def perturbation_audit(model,dataset,device,channels,count_samples,different_reference):
    indices=np.linspace(0,len(dataset)-1,min(count_samples,len(dataset)),dtype=int); x4=np.stack([dataset[int(i)]["input_surface"] for i in indices]); x=select(torch.from_numpy(x4).to(device),channels); _,base=model(x,return_features=True); base=F.normalize(base,dim=1); out={}
    for name,changed in make_perturbations(x).items():
        _,z=model(changed,return_features=True); d=torch.clamp(1-(base*F.normalize(z,dim=1)).sum(1),min=0).cpu().numpy(); reference=np.resize(different_reference,len(d)); out[name]={"same_sample":describe(d),"geometry_different_reference":describe(reference),"same_less_fraction":float((d<reference).mean())}
    return out


class MTARE(Dataset):
    def __init__(self,root): self.files=sorted(Path(root).glob("*.npz"))
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        with np.load(self.files[i],allow_pickle=False) as d: return d["input_surface"].astype(np.float32)


@torch.no_grad()
def mtare_audit(model,root,device,channels):
    ds=MTARE(root); x4=torch.from_numpy(np.stack([ds[i] for i in range(len(ds))])).to(device); x=select(x4,channels); pred,z=model(x,return_features=True); z=F.normalize(z,dim=1); pair=torch.pdist(z).cpu().numpy(); perturb={}
    for name,changed in make_perturbations(x).items():
        _,zp=model(changed,return_features=True); d=torch.clamp(1-(z*F.normalize(zp,dim=1)).sum(1),min=0).cpu().numpy(); perturb[name]=describe(d)
    return {"samples":len(ds),"forward":True,"finite":bool(torch.isfinite(pred).all()),"output_range":[float(pred.min()),float(pred.max())],"output_not_identical":bool(float(pred.flatten(1).std(0).mean())>1e-6),"sample_pair_distance":describe(pair),"perturbations":perturb}


def save_preview(path,x,normal,shuffled,teacher,count):
    fig,axes=plt.subplots(min(count,len(x)),4,figsize=(13,3*min(count,len(x))),squeeze=False)
    for row in range(min(count,len(x))):
        for ax,(im,title) in zip(axes[row],[(x[row,0],"input"),(normal[row,0],"prediction"),(shuffled[row,0],"shuffled latent"),(teacher[row,0],"teacher")]): ax.imshow(im.detach().cpu(),origin="upper",cmap="viridis",vmin=0,vmax=1); ax.set_title(title); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/learning/structural_bottleneck_feasibility.yaml"); p.add_argument("--run-id",default=None); args=p.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text()); run_id=args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"); root=Path(cfg["experiment"]["output_root"])/run_id
    if root.exists(): raise FileExistsError(f"refusing to overwrite {root}")
    names=("config","checkpoint","training_log","completion_metrics","bottleneck_dependency_audit","representation_correlation_audit","matched_pair_audit","perturbation_audit","mtare_inference_audit","previews")
    for name in names: (root/name).mkdir(parents=True,exist_ok=True)
    (root/"config"/"config.yaml").write_text(Path(args.config).read_text()); seed_all(int(cfg["experiment"]["seed"]),bool(cfg["experiment"]["deterministic"])); device=torch.device("cuda"); kwargs={"input_channels":len(cfg["model"]["input_channels"]),"latent_dim":int(cfg["model"]["latent_dim"]),"base_channels":int(cfg["model"]["base_channels"])}; train=StructuralSurfaceDataset(cfg["experiment"]["dataset_dir"],"train"); val=StructuralSurfaceDataset(cfg["experiment"]["dataset_dir"],"val")
    fixed=Subset(train,np.linspace(0,len(train)-1,int(cfg["overfit"]["samples"]),dtype=int).tolist()); fixed_batch=next(iter(DataLoader(fixed,batch_size=int(cfg["overfit"]["batch_size"]),collate_fn=collate))); overfit_model=ForcedGlobalBottleneckNet(**kwargs).to(device); opt=torch.optim.Adam(overfit_model.parameters(),lr=float(cfg["overfit"]["learning_rate"])); g=torch.Generator(device=device).manual_seed(int(cfg["experiment"]["seed"])); overfit=[]
    for _ in range(int(cfg["overfit"]["steps"])): overfit.append(train_step(overfit_model,fixed_batch,opt,device,cfg,g,with_constraints=False)["total"])
    overfit_result={"samples":len(fixed),"steps":len(overfit),"initial_loss":overfit[0],"middle_loss":overfit[len(overfit)//2],"final_loss":overfit[-1],"finite":bool(np.isfinite(overfit).all()),"pass":bool(overfit[-1]<overfit[0]*.4)}; write_json(root/"completion_metrics"/"overfit.json",overfit_result)
    model=ForcedGlobalBottleneckNet(**kwargs).to(device); untrained=ForcedGlobalBottleneckNet(**kwargs).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["train"]["learning_rate"]),weight_decay=float(cfg["train"]["weight_decay"])); train_loader=DataLoader(train,batch_size=int(cfg["train"]["batch_size"]),shuffle=True,num_workers=int(cfg["train"]["num_workers"]),pin_memory=True,collate_fn=collate,generator=torch.Generator().manual_seed(int(cfg["experiment"]["seed"]))); val_loader=DataLoader(val,batch_size=int(cfg["train"]["batch_size"]),shuffle=False,num_workers=int(cfg["train"]["num_workers"]),pin_memory=True,collate_fn=collate); initial_val=evaluate(untrained,val_loader,device,cfg); history=[]; best=float("inf"); generator=torch.Generator(device=device).manual_seed(int(cfg["experiment"]["seed"])+1)
    for epoch in range(1,int(cfg["train"]["epochs"])+1):
        rows=[train_step(model,b,optimizer,device,cfg,generator,True) for b in train_loader]; vm=evaluate(model,val_loader,device,cfg); row={"epoch":epoch,"train_loss":float(np.mean([r["completion"] for r in rows])),"train_total":float(np.mean([r["total"] for r in rows])),"val_loss":vm["loss"],"val_iou":vm["model"]["iou"]}; history.append(row); print(json.dumps(row),flush=True)
        if vm["loss"]<best: best=vm["loss"]; torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"config":cfg},root/"checkpoint"/"best.pt")
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":len(history),"config":cfg},root/"checkpoint"/"last.pt"); ck=torch.load(root/"checkpoint"/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model"]); model.eval(); final=evaluate(model,val_loader,device,cfg); old=json.loads((Path(cfg["experiment"]["comparison_run"])/"completion_metrics"/"summary.json").read_text())["version_b_3ch"]
    completion={"overfit":overfit_result,"untrained_validation":initial_val,"trained_validation":final,"previous_3ch_unet":old,"history":history,"best_epoch":int(ck["epoch"])}; write_json(root/"completion_metrics"/"summary.json",completion)
    first=next(iter(val_loader)); x4=first["input_surface"].to(device); x=select(x4,cfg["model"]["input_channels"]); teacher=first["teacher_surface"].to(device)
    with torch.no_grad(): normal,z=model(x,True); zero=model.decode(torch.zeros_like(z)); shuffled=model.decode(z[torch.roll(torch.arange(len(z),device=device),1)])
    normal_c=count(normal,teacher,float(cfg["audit"]["surface_threshold"])); zero_c=count(zero,teacher,float(cfg["audit"]["surface_threshold"])); shuffle_c=count(shuffled,teacher,float(cfg["audit"]["surface_threshold"])); normal_m=metrics(normal_c); zero_m=metrics(zero_c); shuffle_m=metrics(shuffle_c); zero_mae=float(torch.mean(torch.abs(normal-zero))); shuffle_mae=float(torch.mean(torch.abs(normal-shuffled))); dependency={"normal":normal_m,"zero_latent":zero_m,"shuffled_latent":shuffle_m,"zero_output_mae":zero_mae,"shuffled_output_mae":shuffle_mae,"decoder_inputs":"latent only","high_resolution_skip_paths":0,"gate":"no bypass and shuffled IoU drops by at least 0.05 and zero-latent output MAE exceeds 0.05","pass":bool(normal_m["iou"]-shuffle_m["iou"]>=.05 and zero_mae>.05)}; write_json(root/"bottleneck_dependency_audit"/"summary.json",dependency); save_preview(root/"previews"/"input_prediction_shuffled_teacher.png",x4,normal,shuffled,teacher,int(cfg["audit"]["preview_samples"]))
    features,cell,input_masks,teacher_masks=collect(model,val,device,cfg["model"]["input_channels"]); rng=np.random.default_rng(int(cfg["audit"]["pair_seed"])); left=rng.integers(0,len(val),int(cfg["audit"]["pair_draws"])); right=rng.integers(0,len(val),int(cfg["audit"]["pair_draws"])); keep=left!=right; left,right=left[keep],right[keep]; fd,cell_diff,coverage_shape,geometry=pair_data(features,cell,input_masks,teacher_masks,left,right); corr={"pair_count":int(len(fd)),"feature_vs_surface_cell_count_difference":spearman(fd,cell_diff),"feature_vs_input_coverage_shape_difference":spearman(fd,coverage_shape),"feature_vs_teacher_geometry_difference":spearman(fd,geometry)}; write_json(root/"representation_correlation_audit"/"summary.json",corr)
    geom_low=geometry<=np.percentile(geometry,25); geom_high=geometry>=np.percentile(geometry,75); cover_low=cell_diff<=np.percentile(cell_diff,25); cover_high=cell_diff>=np.percentile(cell_diff,75); class_a=geom_low&cover_high; class_b=geom_high&cover_low; matched={"definition_a":"teacher geometry distance bottom quartile AND surface-cell difference top quartile","definition_b":"surface-cell difference bottom quartile AND teacher geometry distance top quartile","class_a_geometry_similar_coverage_different":{"samples":int(class_a.sum()),"distance":describe(fd[class_a])},"class_b_coverage_similar_geometry_different":{"samples":int(class_b.sum()),"distance":describe(fd[class_b])},"median_gap_b_minus_a":float(np.median(fd[class_b])-np.median(fd[class_a])),"pass":bool(np.median(fd[class_a])<np.median(fd[class_b]))}; write_json(root/"matched_pair_audit"/"summary.json",matched)
    perturb=perturbation_audit(model,val,device,cfg["model"]["input_channels"],int(cfg["audit"]["perturbation_samples"]),fd[class_b]); write_json(root/"perturbation_audit"/"summary.json",perturb); mtare=mtare_audit(model,Path(cfg["experiment"]["dataset_dir"])/"mtare_samples",device,cfg["model"]["input_channels"]); write_json(root/"mtare_inference_audit"/"summary.json",mtare)
    architecture={"input_channels":["surface_mask","mean_height","height_span"],"latent":{"shape":[int(cfg["model"]["latent_dim"])],"fixed_dimension":True},"decoder_source":"linear expansion from latent only","skip_connections":[],"parameter_count":sum(p.numel() for p in model.parameters()),"code_audit":"No encoder tensor or raw input is passed to decode()."}; write_json(root/"bottleneck_dependency_audit"/"architecture.json",architecture)
    teacher_corr=corr["feature_vs_teacher_geometry_difference"]; coverage_dom=max(abs(corr["feature_vs_surface_cell_count_difference"]),abs(corr["feature_vs_input_coverage_shape_difference"])); perturb_pass=all(v["same_sample"]["median"]<v["geometry_different_reference"]["median"] for v in perturb.values()); basic=overfit_result["pass"] and final["model"]["iou"]>initial_val["model"]["iou"]+.1 and final["model"]["iou"]>final["input_copy_baseline"]["iou"]*1.5; full=dependency["pass"] and basic and matched["pass"] and teacher_corr>coverage_dom and perturb_pass and mtare["finite"]
    improved=dependency["pass"] and basic and matched["pass"] and perturb_pass; status="BOTTLENECK_REPRESENTATION_PASS" if full else "BOTTLENECK_REPRESENTATION_MIXED" if improved else "BOTTLENECK_REPRESENTATION_FAIL"; summary={"status":status,"architecture":architecture,"dependency":dependency,"completion":completion,"correlation":corr,"matched_pairs":matched,"perturbation":perturb,"mtare":mtare,"environment":{"hostname":os.uname().nodename,"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)}}; write_json(root/"summary.json",summary); print(json.dumps({"result_dir":str(root),"status":status},indent=2))


if __name__=="__main__": main()
