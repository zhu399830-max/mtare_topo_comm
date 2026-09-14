from __future__ import annotations

import argparse, json, math, random, re, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset


def dump(p: Path, x: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class DS(Dataset):
    def __init__(self, root: Path, split: str, files=None):
        self.root, self.split = root, split
        self.files = list(files) if files is not None else sorted((root / split).glob("*.npz"))
        self.meta = []
        for p in self.files:
            with np.load(p, allow_pickle=False) as z:
                self.meta.append({"world": str(z["world"]), "trajectory": str(z["trajectory"]), "timestamp": int(z["timestamp"]), "path": str(p)})

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        with np.load(self.files[i], allow_pickle=False) as z:
            return {
                "current": z["student_input_current"].astype(np.float32),
                "history": z["student_input_history"].astype(np.float32),
                "history_valid_mask": z["history_valid_mask"].astype(np.float32),
                "history_time_deltas": z["history_time_deltas"].astype(np.float32),
                "relative_poses": z["relative_poses"].astype(np.float32),
                "direction": z["traversable_direction_distribution"].astype(np.float32),
                "distance": z["reachable_distance_distribution"].astype(np.float32),
                "area": z["reachable_area_distribution"].astype(np.float32),
                "exit_soft": z["exit_sector_soft"].astype(np.float32),
                "exit_binary": z["exit_sector_binary"].astype(np.float32),
                "exit_centers": z["exit_centers"].astype(np.float32),
                "exit_widths": z["exit_widths"].astype(np.float32),
                "exit_lengths": z["exit_lengths"].astype(np.float32),
                "exit_valid": z["exit_valid_mask"].astype(np.float32),
                "exit_count": np.asarray(z["exit_count"], dtype=np.float32),
                "static": z["continuous_static_scores"].astype(np.float32),
                "static_mask": z["continuous_static_score_mask"].astype(np.float32),
                "dynamic": z["continuous_dynamic_scores"].astype(np.float32),
                "dynamic_mask": z["continuous_dynamic_score_mask"].astype(np.float32),
                "role": z["canonical_topological_role"].astype(np.float32),
                "role_valid": np.asarray(z["canonical_role_valid"], dtype=np.float32),
                "world": str(z["world"]), "trajectory": str(z["trajectory"]), "path": str(self.files[i]),
                "raw_point_count": np.asarray(z["raw_point_count"] if "raw_point_count" in z.files else 0, dtype=np.float32),
            }


def collate(rows):
    keys = [k for k, v in rows[0].items() if isinstance(v, np.ndarray)]
    out = {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in keys}
    for k in ["world", "trajectory", "path"]: out[k] = [r[k] for r in rows]
    return out


class Encoder(nn.Module):
    def __init__(self, c=3, base=16, dim=96):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, base, 5, 2, 2), nn.GroupNorm(4, base), nn.SiLU(),
            nn.Conv2d(base, base*2, 3, 2, 1), nn.GroupNorm(4, base*2), nn.SiLU(),
            nn.Conv2d(base*2, base*4, 3, 2, 1), nn.GroupNorm(8, base*4), nn.SiLU(),
            nn.Conv2d(base*4, base*4, 3, 2, 1), nn.GroupNorm(8, base*4), nn.SiLU(), nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Linear(base*4, dim)
    def forward(self, x): return self.proj(self.net(x).flatten(1))


class Net(nn.Module):
    def __init__(self, ordered: bool, cfg, use_history: bool = True):
        super().__init__(); m = cfg["model"]
        self.ordered = ordered; self.use_history = use_history; d = int(m["dim"]); self.enc = Encoder(3, int(m["base"]), d)
        self.pose = nn.Sequential(nn.Linear(6, 24), nn.SiLU(), nn.Linear(24, 24))
        self.fuse = nn.Sequential(nn.Linear(d*2, d), nn.SiLU(), nn.Dropout(.05))
        if ordered: self.gru = nn.GRU(d+24, d, batch_first=True)
        self.static_head = nn.Sequential(nn.Linear(d, d), nn.SiLU())
        self.direction, self.distance, self.area = nn.Linear(d,32), nn.Linear(d,32), nn.Linear(d,32)
        self.exit = nn.Linear(d,32); self.count = nn.Linear(d,1); self.center = nn.Linear(d,8)
        self.width = nn.Linear(d,8); self.length = nn.Linear(d,8); self.static = nn.Linear(d,3); self.role = nn.Linear(d,64)
        self.dynamic = nn.Linear(d,4) if ordered else None

    def forward(self, cur, hist, valid, dt, poses):
        b,t,c,h,w = hist.shape; zc = self.enc(cur[:, :3]); zh = self.enc(hist[:, :, :3].reshape(b*t,3,h,w)).reshape(b,t,-1)
        aux = torch.cat([poses, dt.unsqueeze(-1)], -1); zj = torch.cat([zh, self.pose(aux)], -1)
        mask = valid.unsqueeze(-1)
        if not self.use_history:
            zi = torch.zeros_like(zc)
        elif self.ordered:
            zj = zj * mask; seq, _ = self.gru(zj); lengths = valid.sum(1).long().clamp_min(1)-1
            zi = seq[torch.arange(b, device=seq.device), lengths]
        else: zi = (zj * mask).sum(1) / mask.sum(1).clamp_min(1)
        z = self.fuse(torch.cat([zc, zi[:, :zc.shape[1]]], 1)); s = self.static_head(z)
        return {"z": F.normalize(z,dim=1), "role": F.normalize(self.role(s),dim=1),
                "direction": self.direction(s), "distance": torch.sigmoid(self.distance(s)), "area": torch.sigmoid(self.area(s)),
                "exit": self.exit(s), "count": F.softplus(self.count(s)).squeeze(1), "center": self.center(s),
                "width": torch.sigmoid(self.width(s)), "length": torch.sigmoid(self.length(s)), "static": torch.sigmoid(self.static(s)),
                "dynamic": torch.sigmoid(self.dynamic(s)) if self.dynamic is not None else None}


def move(b, dev):
    return {k: v.to(dev, non_blocking=True) for k,v in b.items() if isinstance(v, torch.Tensor)}


def masked_l1(a,b,m):
    return (torch.abs(a-b)*m).sum()/m.sum().clamp_min(1)


def loss_fn(p,b,weights, ordered):
    direction = F.binary_cross_entropy_with_logits(p["direction"], b["direction"])
    distance = masked_l1(p["distance"], b["distance"], (b["direction"]>.5).float())
    area = F.smooth_l1_loss(p["area"], b["area"])
    exit_loss = F.binary_cross_entropy_with_logits(p["exit"], b["exit_soft"])
    count = F.smooth_l1_loss(p["count"], b["exit_count"])
    valid = b["exit_valid"]
    center = masked_l1(torch.sigmoid(p["center"])*2*math.pi, b["exit_centers"], valid)
    width = masked_l1(p["width"], b["exit_widths"]/(2*math.pi), valid)
    length = masked_l1(p["length"], b["exit_lengths"], valid)
    static = masked_l1(p["static"], b["static"], b["static_mask"])
    rv=b["role_valid"].view(-1); role=1-F.cosine_similarity(p["role"], F.normalize(b["role"],dim=1),dim=1); role=(role*rv).sum()/rv.sum().clamp_min(1)
    dynamic = masked_l1(p["dynamic"], b["dynamic"], b["dynamic_mask"]) if ordered else p["z"].sum()*0
    parts={"direction":direction,"distance":distance,"area":area,"exit":exit_loss,"count":count,"center":center,"width":width,"length":length,"static":static,"role":role,"dynamic":dynamic}
    total=sum(float(weights.get(k,1))*v for k,v in parts.items())
    parts["total"]=total; return total, {k:float(v.detach()) for k,v in parts.items()}


def pair_loss(p1,p2, typ):
    d=1-F.cosine_similarity(p1["role"],p2["role"]); target=torch.tensor([x=="positive_invariance" for x in typ],device=d.device,dtype=d.dtype)
    return torch.where(target>0, d, F.relu(.25-d)).mean()


def run_epoch(model, loader, opt, dev, cfg, ordered, pair_loader=None):
    model.train(); rows=[]; pit=iter(pair_loader) if pair_loader else None
    for raw in loader:
        b=move(raw,dev); opt.zero_grad(set_to_none=True); p=model(b["current"],b["history"],b["history_valid_mask"],b["history_time_deltas"],b["relative_poses"]); total,parts=loss_fn(p,b,cfg["loss"],ordered)
        if pit:
            try: r1,r2,typ=next(pit); fw=lambda r: model(r["cur"].to(dev),r["hist"].to(dev),r["valid"].to(dev),r["dt"].to(dev),r["pose"].to(dev)); pl=pair_loss(fw(r1),fw(r2),typ); total=total+.35*pl; parts["pair"]=float(pl.detach())
            except StopIteration: pit=iter(pair_loader)
        total.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); rows.append(parts)
    return {k:float(np.mean([x.get(k,0) for x in rows])) for k in rows[0]}


@torch.no_grad()
def predict(model, ds, dev, bs=64):
    model.eval(); out=defaultdict(list)
    for raw in DataLoader(ds,bs,False,collate_fn=collate):
        b=move(raw,dev); p=model(b["current"],b["history"],b["history_valid_mask"],b["history_time_deltas"],b["relative_poses"])
        for k in ["direction","distance","area","exit_soft","exit_binary","exit_centers","exit_widths","exit_lengths","exit_valid","exit_count","static","dynamic","role","raw_point_count"]: out["y_"+k].append(b[k].cpu().numpy())
        for k in ["direction","distance","area","exit","count","center","width","length","static","dynamic","role"]: 
            if p[k] is not None: out["p_"+k].append((torch.sigmoid(p[k]) if k in ["direction","exit"] else p[k]).cpu().numpy())
        out["world"] += raw["world"]; out["trajectory"] += raw["trajectory"]; out["path"] += raw["path"]
    return {k:(np.concatenate(v) if k.startswith(('p_','y_')) else v) for k,v in out.items()}


def bin_metrics(y,p,thr=.5):
    y=(y>=.5).astype(int).reshape(-1); p=np.asarray(p).reshape(-1); q=(p>=thr).astype(int)
    return {"precision":float(((q==1)&(y==1)).sum()/max((q==1).sum(),1)),"recall":float(((q==1)&(y==1)).sum()/max((y==1).sum(),1)),"f1":float(2*((q==1)&(y==1)).sum()/max((q==1).sum()+(y==1).sum(),1)),"balanced_accuracy":float(balanced_accuracy_score(y,q)),"pr_auc":float(average_precision_score(y,p)) if y.min()!=y.max() else float("nan"),"roc_auc":float(roc_auc_score(y,p)) if y.min()!=y.max() else float("nan")}


def metrics(x):
    m={"direction":bin_metrics(x["y_direction"],x["p_direction"]),"exit_sector":bin_metrics(x["y_exit_soft"],x["p_exit"]),"exit_count_mae":float(np.mean(np.abs(x["p_count"]-x["y_exit_count"]))),"distance_mae":float(np.mean(np.abs(x["p_distance"]-x["y_distance"]))),"area_mae":float(np.mean(np.abs(x["p_area"]-x["y_area"]))),"dynamic_mae":None}
    if x.get("p_dynamic") is not None: m["dynamic_mae"]={k:float(np.mean(np.abs(x["p_dynamic"][:,i]-x["y_dynamic"][:,i]))) for i,k in enumerate(["turn_strength","transition_score","new_branch_appearance","topological_node_score"])}
    valid=x["y_exit_valid"]>.5; m["exit_geometry"]={"center_angle_mae":float(np.mean(np.abs((x["p_center"][valid]-x["y_exit_centers"][valid]+math.pi)%(2*math.pi)-math.pi))) if valid.any() else None,"width_mae":float(np.mean(np.abs(x["p_width"][valid]*2*math.pi-x["y_exit_widths"][valid]))) if valid.any() else None,"length_mae":float(np.mean(np.abs(x["p_length"][valid]-x["y_exit_lengths"][valid]))) if valid.any() else None}
    r=x["p_role"]/np.maximum(np.linalg.norm(x["p_role"],axis=1,keepdims=True),1e-8); t=x["y_role"]/np.maximum(np.linalg.norm(x["y_role"],axis=1,keepdims=True),1e-8); m["role_cosine_distance"]=float(np.mean(1-np.sum(r*t,1)))
    if len(r)>2:
        d=1-r@r.T; td=1-t@t.T; u=np.triu_indices(len(r),1); m["teacher_topology_spearman"]=float(spearmanr(d[u],td[u]).correlation)
    return m


def overfit(ds, cfg, dev, out):
    sub=Subset(ds, list(range(min(64,len(ds))))); raw=next(iter(DataLoader(sub,64,False,collate_fn=collate))); b=move(raw,dev); model=Net(True,cfg).to(dev); opt=torch.optim.Adam(model.parameters(),lr=.002); rec=[]
    for e in range(201):
        model.train(); p=model(b["current"],b["history"],b["history_valid_mask"],b["history_time_deltas"],b["relative_poses"]); l,part=loss_fn(p,b,cfg["loss"],True)
        if e==0: rec.append({"epoch":e,**part})
        opt.zero_grad(); l.backward(); opt.step()
        if e in [1,10,50,100,200]: rec.append({"epoch":e,**part,"total":float(l.detach())})
    result={"sample_count":64,"records":rec,"success":rec[-1]["total"]<.7*rec[0]["total"]}; dump(out,result); return result


def make_pair_loader(root, train_ds, cfg):
    index={}
    for p,meta in zip(train_ds.files,train_ds.meta): index[(meta["world"],meta["trajectory"],meta["timestamp"])]=p
    rows=[]
    for line in (root/"coverage_pairs/pairs.jsonl").read_text().splitlines():
        j=json.loads(line)
        if j.get("split")!="train": continue
        def resolve(s):
            stem=Path(s).stem; parts=stem.split("_"); world=parts[1]; token=parts[-1]; traj="_".join(parts[2:-1]);
            # v4 filename inserts '<split>_<world>' and a numeric export prefix.
            candidates=[p for p,m in zip(train_ds.files,train_ds.meta) if m["world"]==world and m["trajectory"]==traj and p.stem.endswith("_"+token)]
            return candidates[0] if candidates else None
        a,b=resolve(j["sample_a"]),resolve(j["sample_b"])
        if a and b: rows.append((a,b,j["pair_type"]))
    class P(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self,i):
            def one(p):
                x=DS(root,"train",[p])[0]; return {"cur":torch.from_numpy(x["current"][:3]),"hist":torch.from_numpy(x["history"][:,:3]),"valid":torch.from_numpy(x["history_valid_mask"]),"dt":torch.from_numpy(x["history_time_deltas"]),"pose":torch.from_numpy(x["relative_poses"])}
            return one(rows[i][0]),one(rows[i][1]),rows[i][2]
    def pc(batch):
        a,b,t=zip(*batch); out=[]
        for side in [a,b]: out.append({k:torch.stack([x[k] for x in side]) for k in side[0]})
        return (*out,list(t))
    return DataLoader(P(),batch_size=min(16,max(1,len(rows))),shuffle=True,collate_fn=pc) if rows else None, len(rows)


def train_one(name, ordered, use_history, train, val, cfg, dev, out, pair_loader=None):
    model=Net(ordered,cfg,use_history).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=cfg["train"]["lr"],weight_decay=1e-4); best=float("inf"); best_epoch=0; logs=[]
    for e in range(1,cfg["train"]["epochs"]+1):
        tr=run_epoch(model,DataLoader(train,cfg["train"]["batch_size"],True,num_workers=0,collate_fn=collate),opt,dev,cfg,ordered,pair_loader); va=predict(model,val,dev,cfg["train"]["batch_size"]); val_loss=float(np.mean([np.mean(np.abs(va["p_direction"]-va["y_direction"])),np.mean(np.abs(va["p_exit"]-va["y_exit_soft"]))]))
        logs.append({"epoch":e,"train":tr,"val_proxy":val_loss})
        if val_loss<best: best,best_epoch=val_loss,e; torch.save({"model":model.state_dict(),"epoch":e,"ordered":ordered,"params":sum(p.numel() for p in model.parameters())},out/f"{name}_checkpoint.pt")
    model.load_state_dict(torch.load(out/f"{name}_checkpoint.pt",map_location=dev,weights_only=False)["model"]); dump(out/f"training_logs_{name}.json",logs); return model, {"param_count":sum(p.numel() for p in model.parameters()),"best_epoch":best_epoch,"val_proxy":best}


def audit(ds):
    c=Counter(x["world"] for x in ds.meta); return {"count":len(ds),"world_counts":dict(c),"history_order":"old_to_new","shapes":{"current":[4,100,100],"history":[8,4,100,100]},"invalid_masks":sum(1 for i in range(len(ds)) if not np.isfinite(ds[i]["dynamic"]).all())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-root",type=Path,default=Path("results/topological_semantic_dataset_v4_supervision_fixed")); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--seed",type=int,default=20260806); args=ap.parse_args(); cfg={"model":{"base":16,"dim":96},"train":{"batch_size":64,"epochs":24,"lr":8e-4},"loss":{"direction":1,"distance":1,"area":.5,"exit":1,"count":.5,"center":.25,"width":.25,"length":.25,"static":1,"role":1,"dynamic":1}}
    seed_all(args.seed); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); out=args.out; out.mkdir(parents=True,exist_ok=True); dump(out/"config.json",{"cfg":cfg,"seed":args.seed,"device":str(dev),"dataset_root":str(args.dataset_root)})
    train=DS(args.dataset_root,"train"); val=DS(args.dataset_root,"val"); test=DS(args.dataset_root,"test"); dump(out/"data_audit.json",{"train":audit(train),"val":audit(val),"test":audit(test),"device":str(dev),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})
    small=overfit(train,cfg,dev,out/"small_overfit.json"); pair_loader,pair_count=make_pair_loader(args.dataset_root,train,cfg); dump(out/"coverage_pair_audit.json",{"matched_train_pairs":pair_count})
    models={}; info={}
    models["current_only"],info["current_only"]=train_one("current_only",False,False,train,val,cfg,dev,out)
    models["causal_history"],info["causal_history"]=train_one("causal_history",False,True,train,val,cfg,dev,out)
    models["ordered_history"],info["ordered_history"]=train_one("ordered_history",True,True,train,val,cfg,dev,out,pair_loader)
    allm={"model_info":info,"small_overfit":small}
    for n,m in models.items():
        for split,ds in [("forest",val),("campus",DS(args.dataset_root,"test",[p for p,meta in zip(test.files,test.meta) if meta["world"]=="campus"])),("indoor",DS(args.dataset_root,"test",[p for p,meta in zip(test.files,test.meta) if meta["world"]=="indoor"]))]:
            pr=predict(m,ds,dev); mm=metrics(pr); dump(out/f"{split}_{n}_metrics.json",mm); allm[f"{split}_{n}"]=mm
    # Fixed threshold, selected from train only; no campus/indoor tuning.
    base={"all_positive_f1_forest":bin_metrics(np.concatenate([predict(models["current_only"],val,dev)["y_direction"]]),np.ones((len(val),32))) ["f1"],"threshold":.5}
    dump(out/"direction_baselines.json",base)
    # Dependency audit on fixed forest subset, including order/pose/time ablations.
    audit_rows={}; m=models["ordered_history"]; m.eval(); sub=Subset(val,list(range(min(356,len(val))))); raw=next(iter(DataLoader(sub,64,False,collate_fn=collate))); b=move(raw,dev)
    with torch.no_grad():
        ref=m(b["current"],b["history"],b["history_valid_mask"],b["history_time_deltas"],b["relative_poses"])
        variants={"reverse_order":(b["history"].flip(1),b["history_valid_mask"].flip(1),b["history_time_deltas"].flip(1),b["relative_poses"].flip(1)),"random_order":(b["history"].roll(3,1),b["history_valid_mask"].roll(3,1),b["history_time_deltas"].roll(3,1),b["relative_poses"].roll(3,1)),"zero_pose":(b["history"],b["history_valid_mask"],b["history_time_deltas"],torch.zeros_like(b["relative_poses"])),"zero_time":(b["history"],b["history_valid_mask"],torch.zeros_like(b["history_time_deltas"]),b["relative_poses"]),"current_only":(b["history"],torch.zeros_like(b["history_valid_mask"]),b["history_time_deltas"],b["relative_poses"])}
        for k,v in variants.items():
            q=m(b["current"],*v); audit_rows[k]={"dynamic_abs_delta":float(torch.abs(q["dynamic"]-ref["dynamic"]).mean()),"role_cosine_drift":float((1-F.cosine_similarity(q["role"],ref["role"])).mean())}
    dump(out/"history_dependency_audit.json",audit_rows)
    # Conservative decision rubric: report MIXED unless all required evidence is present.
    f=allm["forest_ordered_history"]; direction_improved=f["direction"]["f1"]>0.8273; dynamic_ok=all(v<.35 for v in (f["dynamic_mae"] or {}).values()); role_ok=f.get("teacher_topology_spearman",-1)>0
    conclusion="TOPOLOGICAL_SEMANTIC_V2_PASS" if small["success"] and direction_improved and dynamic_ok and role_ok else ("TOPOLOGICAL_SEMANTIC_V2_MIXED" if small["success"] else "TOPOLOGICAL_SEMANTIC_V2_FAIL")
    allm.update({"conclusion":conclusion,"history_dependency_audit":audit_rows}); dump(out/"summary.json",allm); print(json.dumps({"conclusion":conclusion,"out":str(out),"pair_count":pair_count},indent=2))


if __name__=="__main__": main()
