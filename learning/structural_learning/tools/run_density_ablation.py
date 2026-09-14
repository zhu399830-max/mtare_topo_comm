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
from torch.utils.data import DataLoader

from learning.structural_learning.dataset import StructuralSurfaceDataset
from learning.structural_learning.feasibility_model import TinySurfaceCompletionNet, completion_loss


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def seed_all(seed: int, deterministic: bool) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def collate(samples):
    return {"input_surface": torch.from_numpy(np.stack([s["input_surface"] for s in samples])), "teacher_surface": torch.from_numpy(np.stack([s["teacher_surface"] for s in samples]))}


def select_channels(x: torch.Tensor, channels: list[int]) -> torch.Tensor:
    return x[:, channels]


def counts(prediction, teacher, threshold):
    pred, target = prediction[:, 0] >= threshold, teacher[:, 0] >= 0.5
    return np.asarray([float((pred & target).sum()), float((pred & ~target).sum()), float((~pred & target).sum())])


def count_metrics(value):
    tp, fp, fn = value
    return {"surface_iou": float(tp / max(tp + fp + fn, 1)), "surface_dice": float(2 * tp / max(2 * tp + fp + fn, 1)), "surface_precision": float(tp / max(tp + fp, 1)), "surface_recall": float(tp / max(tp + fn, 1))}


@torch.no_grad()
def evaluate(model, loader, channels, device, threshold, positive_weight):
    model.eval(); losses=[]; model_counts=np.zeros(3); baseline_counts=np.zeros(3)
    for batch in loader:
        x4=batch["input_surface"].to(device); teacher=batch["teacher_surface"].to(device); x=select_channels(x4, channels)
        pred=model(x); loss,_=completion_loss(pred,teacher,positive_weight); losses.append(float(loss))
        model_counts += counts(pred,teacher,threshold); baseline_counts += counts(x4,teacher,threshold)
    return {"loss":float(np.mean(losses)),"model":count_metrics(model_counts),"input_copy_baseline":count_metrics(baseline_counts)}


def train_step(model,batch,channels,optimizer,device,positive_weight):
    model.train(); x=select_channels(batch["input_surface"].to(device),channels); teacher=batch["teacher_surface"].to(device)
    optimizer.zero_grad(set_to_none=True); prediction=model(x); loss,_=completion_loss(prediction,teacher,positive_weight)
    if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
    loss.backward(); optimizer.step(); return float(loss.detach())


def rank(values):
    order=np.argsort(values,kind="mergesort"); result=np.empty_like(order,dtype=np.float64); result[order]=np.arange(len(values)); return result


def spearman(a,b): return float(np.corrcoef(rank(a),rank(b))[0,1])


def describe(values):
    a=np.asarray(values,dtype=np.float64)
    return {"count":int(len(a)),"mean":float(a.mean()),"median":float(np.median(a)),"p10":float(np.percentile(a,10)),"p90":float(np.percentile(a,90)),"max":float(a.max())}


@torch.no_grad()
def collect(model_a,model_b,dataset,device):
    fa,fb,counts_in,density,teacher=[],[],[],[],[]
    for start in range(0,len(dataset),32):
        samples=[dataset[i] for i in range(start,min(start+32,len(dataset)))]
        x4_np=np.stack([s["input_surface"] for s in samples]); x4=torch.from_numpy(x4_np).to(device)
        _,za=model_a(x4,return_features=True); _,zb=model_b(x4[:,[0,2,3]],return_features=True)
        fa.append(torch.nn.functional.normalize(za,dim=1).cpu().numpy()); fb.append(torch.nn.functional.normalize(zb,dim=1).cpu().numpy())
        mask=x4_np[:,0]>0.5; counts_in.extend(mask.sum(axis=(1,2)).tolist()); density.extend([float(x4_np[i,1][mask[i]].mean()) for i in range(len(samples))]); teacher.extend([(s["teacher_surface"][0]>0.5).reshape(-1) for s in samples])
    return np.concatenate(fa),np.concatenate(fb),np.asarray(counts_in),np.asarray(density),np.stack(teacher)


def shared_pairs(size,draws,seed):
    rng=np.random.default_rng(seed); left=rng.integers(0,size,size=draws); right=rng.integers(0,size,size=draws); valid=left!=right; return left[valid],right[valid]


def pair_quantities(features,left,right,counts_in,density,teacher):
    fd=1-np.sum(features[left]*features[right],axis=1)
    count_diff=np.abs(counts_in[left]-counts_in[right])/np.maximum(np.maximum(counts_in[left],counts_in[right]),1)
    density_diff=np.abs(density[left]-density[right])
    inter=np.logical_and(teacher[left],teacher[right]).sum(axis=1); union=np.logical_or(teacher[left],teacher[right]).sum(axis=1); geom=1-inter/np.maximum(union,1)
    return fd,count_diff,density_diff,geom


@torch.no_grad()
def density_perturbation(model_a,model_b,dataset,device,count,seed):
    indices=np.linspace(0,len(dataset)-1,min(count,len(dataset)),dtype=int); x4=torch.from_numpy(np.stack([dataset[int(i)]["input_surface"] for i in indices])).to(device)
    _,base_a=model_a(x4,return_features=True); _,base_b=model_b(x4[:,[0,2,3]],return_features=True)
    base_a=torch.nn.functional.normalize(base_a,dim=1); base_b=torch.nn.functional.normalize(base_b,dim=1)
    generator=torch.Generator(device=device).manual_seed(seed); variants={}
    random_drop=x4.clone(); scale=torch.empty_like(random_drop[:,1]).uniform_(0.1,0.7,generator=generator); random_drop[:,1]*=scale; variants["random_density_reduction"]=random_drop
    local=x4.clone(); local[:,1,20:80,20:80]*=0.25; variants["local_density_scaling"]=local
    nonuniform=x4.clone(); gradient=torch.linspace(0.15,1.0,x4.shape[-1],device=device)[None,None,:]; nonuniform[:,1]*=gradient; variants["nonuniform_density_degradation"]=nonuniform
    output={}
    for name,changed in variants.items():
        _,za=model_a(changed,return_features=True); _,zb=model_b(changed[:,[0,2,3]],return_features=True)
        za=torch.nn.functional.normalize(za,dim=1); zb=torch.nn.functional.normalize(zb,dim=1)
        da=np.maximum((1-(base_a*za).sum(dim=1)).cpu().numpy(),0.0); db=np.maximum((1-(base_b*zb).sum(dim=1)).cpu().numpy(),0.0)
        output[name]={"version_a":describe(da),"version_b":describe(db)}
    return output


def save_preview(path,x,pred_a,pred_b,teacher,count):
    fig,axes=plt.subplots(min(count,len(x)),4,figsize=(13,3*min(count,len(x))),squeeze=False)
    for row in range(min(count,len(x))):
        for ax,(image,title) in zip(axes[row],[(x[row,0],"input"),(pred_a[row,0],"4ch A"),(pred_b[row,0],"3ch B"),(teacher[row,0],"teacher")]):
            ax.imshow(image.detach().cpu(),origin="upper",cmap="viridis",vmin=0,vmax=1); ax.set_title(title); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/learning/structural_density_ablation.yaml"); parser.add_argument("--run-id",default=None); args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); run_id=args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"); root=Path(cfg["experiment"]["output_root"])/run_id
    if root.exists(): raise FileExistsError(f"refusing to overwrite {root}")
    for name in ("config","checkpoint","training_log","completion_metrics","representation_audit","density_perturbation","pairwise_geometry_audit","skip_connection_audit","previews"): (root/name).mkdir(parents=True,exist_ok=True)
    (root/"config"/"config.yaml").write_text(Path(args.config).read_text())
    seed_all(int(cfg["experiment"]["seed"]),bool(cfg["experiment"]["deterministic"])); device=torch.device("cuda"); train=StructuralSurfaceDataset(cfg["experiment"]["dataset_dir"],"train"); val=StructuralSurfaceDataset(cfg["experiment"]["dataset_dir"],"val")
    channels_a=list(cfg["model"]["version_a_channels"]); channels_b=list(cfg["model"]["version_b_channels"]); common={"base_channels":int(cfg["model"]["base_channels"]),"latent_dim":int(cfg["model"]["latent_dim"])}
    model_a=TinySurfaceCompletionNet(**common,input_channels=4).to(device); baseline=torch.load(Path(cfg["experiment"]["baseline_run"])/"checkpoint"/"best.pt",map_location=device,weights_only=False); model_a.load_state_dict(baseline["model"]); model_a.eval()
    model_b=TinySurfaceCompletionNet(**common,input_channels=3).to(device); optimizer=torch.optim.AdamW(model_b.parameters(),lr=float(cfg["train"]["learning_rate"]),weight_decay=float(cfg["train"]["weight_decay"])); positive=float(cfg["loss"]["positive_weight"])
    train_loader=DataLoader(train,batch_size=int(cfg["train"]["batch_size"]),shuffle=True,num_workers=int(cfg["train"]["num_workers"]),pin_memory=True,collate_fn=collate,generator=torch.Generator().manual_seed(int(cfg["experiment"]["seed"])))
    val_loader=DataLoader(val,batch_size=int(cfg["train"]["batch_size"]),shuffle=False,num_workers=int(cfg["train"]["num_workers"]),pin_memory=True,collate_fn=collate); history=[]; best=float("inf")
    for epoch in range(1,int(cfg["train"]["epochs"])+1):
        losses=[train_step(model_b,batch,channels_b,optimizer,device,positive) for batch in train_loader]; vm=evaluate(model_b,val_loader,channels_b,device,float(cfg["evaluation"]["surface_threshold"]),positive); row={"epoch":epoch,"train_loss":float(np.mean(losses)),"val_loss":vm["loss"],"val_iou":vm["model"]["surface_iou"]}; history.append(row); print(json.dumps(row),flush=True)
        if vm["loss"]<best: best=vm["loss"]; torch.save({"model":model_b.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"config":cfg},root/"checkpoint"/"best_3ch.pt")
    torch.save({"model":model_b.state_dict(),"optimizer":optimizer.state_dict(),"epoch":len(history),"config":cfg},root/"checkpoint"/"last_3ch.pt"); saved=torch.load(root/"checkpoint"/"best_3ch.pt",map_location=device,weights_only=False); model_b.load_state_dict(saved["model"]); model_b.eval()
    metric_a=evaluate(model_a,val_loader,channels_a,device,float(cfg["evaluation"]["surface_threshold"]),positive); metric_b=evaluate(model_b,val_loader,channels_b,device,float(cfg["evaluation"]["surface_threshold"]),positive)
    completion={"version_a_4ch":metric_a,"version_b_3ch":metric_b,"history_b":history,"best_epoch_b":int(saved["epoch"]),"parameters_a":sum(p.numel() for p in model_a.parameters()),"parameters_b":sum(p.numel() for p in model_b.parameters())}; write_json(root/"completion_metrics"/"summary.json",completion)
    fa,fb,cin,density,teacher=collect(model_a,model_b,val,device); left,right=shared_pairs(len(val),int(cfg["evaluation"]["pair_draws"]),int(cfg["evaluation"]["pair_seed"])); fda,count_diff,density_diff,geom=pair_quantities(fa,left,right,cin,density,teacher); fdb,_,_,_=pair_quantities(fb,left,right,cin,density,teacher)
    representation={"pair_seed":int(cfg["evaluation"]["pair_seed"]),"pair_count":int(len(left)),"version_a":{"feature_vs_density_spearman":spearman(fda,density_diff),"feature_vs_teacher_geometry_spearman":spearman(fda,geom)},"version_b":{"feature_vs_density_spearman":spearman(fdb,density_diff),"feature_vs_teacher_geometry_spearman":spearman(fdb,geom)}}; write_json(root/"representation_audit"/"summary.json",representation)
    perturb=density_perturbation(model_a,model_b,val,device,int(cfg["evaluation"]["density_perturbation_samples"]),int(cfg["evaluation"]["pair_seed"])); write_json(root/"density_perturbation"/"summary.json",perturb)
    density_low=density_diff<=np.percentile(density_diff,25); density_high=density_diff>=np.percentile(density_diff,75); geom_low=geom<=np.percentile(geom,25); geom_high=geom>=np.percentile(geom,75); different=density_low&geom_high; similar=density_high&geom_low
    pair_audit={"selection":"shared pair set; quartile intersections","similar_density_geometry_different":{"count":int(different.sum()),"version_a":describe(fda[different]),"version_b":describe(fdb[different])},"density_different_geometry_similar":{"count":int(similar.sum()),"version_a":describe(fda[similar]),"version_b":describe(fdb[similar])}}; write_json(root/"pairwise_geometry_audit"/"summary.json",pair_audit)
    skip={"enc1_skip":{"shape":"[B,16,100,100]","decoder_use":"concatenated after final upsample"},"enc2_skip":{"shape":"[B,32,50,50]","decoder_use":"concatenated after bottleneck upsample"},"bottleneck":{"shape":"[B,64,25,25]","decoder_use":"upsampled into dec2","reported_latent":"global average pooling produces [B,64] only for representation audit"},"bypass_risk":"Yes. Both high-resolution skip paths reach the prediction head, so teacher prediction is not constrained to pass only through the pooled 64-D representation.","third_model_trained":False}; write_json(root/"skip_connection_audit"/"summary.json",skip)
    batch=next(iter(val_loader)); x4=batch["input_surface"].to(device); teacher_batch=batch["teacher_surface"].to(device)
    with torch.no_grad(): pa=model_a(x4); pb=model_b(x4[:,channels_b])
    save_preview(root/"previews"/"input_a_b_teacher.png",x4,pa,pb,teacher_batch,int(cfg["evaluation"]["preview_samples"]))
    density_drop=representation["version_a"]["feature_vs_density_spearman"]-representation["version_b"]["feature_vs_density_spearman"]; geometry_change=representation["version_b"]["feature_vs_teacher_geometry_spearman"]-representation["version_a"]["feature_vs_teacher_geometry_spearman"]
    b_beats=metric_b["model"]["surface_iou"]>metric_b["input_copy_baseline"]["surface_iou"]*1.5; perturb_b=max(v["version_b"]["median"] for v in perturb.values()); perturb_a=np.mean([v["version_a"]["median"] for v in perturb.values()]); status="DENSITY_ABLATION_PASS" if b_beats and density_drop>0.1 and geometry_change>=0 and perturb_b<perturb_a*0.5 else "DENSITY_ABLATION_MIXED" if b_beats and density_drop>0 else "DENSITY_ABLATION_FAIL"
    summary={"status":status,"completion":completion,"representation":representation,"density_perturbation":perturb,"pairwise_geometry":pair_audit,"skip_connection_audit":skip,"density_correlation_drop":density_drop,"geometry_correlation_change":geometry_change,"environment":{"hostname":os.uname().nodename,"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)}}; write_json(root/"summary.json",summary); print(json.dumps({"result_dir":str(root),"status":status},indent=2))


if __name__=="__main__": main()
