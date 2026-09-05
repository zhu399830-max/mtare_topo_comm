#!/usr/bin/env python3
"""Train one initialized local forward/lateral/up event-center head."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from mtare_topo.data.gse_action_set_cache import ActionSetNodeDataset
from mtare_topo.representation.gse_action_set_node import ActionSetNodeDetector
from mtare_topo.representation.gse_event_center_offset import EventCenterVectorHead
from mtare_topo.teacher.gse_event_center_teacher import local_event_center_vectors
from train_gse_event_center_pair_consistency_v1 import _collate, _groups, _sample_batch


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _infer(base, head, dataset, count, device):
    import torch
    output = []; base.eval(); head.eval()
    with torch.inference_mode():
        for start in range(0, count, 512):
            local = np.arange(start, min(count, start + 512)); tokens, mask = _collate(dataset, local)
            context = base(torch.from_numpy(tokens).to(device), torch.from_numpy(mask).to(device))["causal_context"]
            output.append(head(context).cpu().numpy())
    return np.concatenate(output)


def _global_centers(vector, rows, sensor, basis):
    return sensor[rows] + np.einsum("nij,ni->nj", basis[rows], vector)


def _cross_metrics(center, rows, identity, traversal, target_center):
    if target_center.shape != center.shape:
        raise ValueError("cross-view target centers must align with selected rows")
    grouped = defaultdict(lambda: defaultdict(list))
    for local, row in enumerate(rows): grouped[str(identity[row])][str(traversal[row])].append(local)
    macro_error, macro_within = [], []
    for views in grouped.values():
        names = sorted(views); errors, within = [], []
        for left_index, left_name in enumerate(names):
            left = np.asarray(views[left_name], dtype=np.int64)
            for right_name in names[left_index + 1:]:
                right = np.asarray(views[right_name], dtype=np.int64)
                predicted_delta = (center[left, None] - center[right[None, :]]).reshape(-1, 3)
                target_delta = (target_center[left, None] - target_center[right[None, :]]).reshape(-1, 3)
                errors.extend(np.linalg.norm(predicted_delta - target_delta, axis=1).tolist())
                within.extend((np.linalg.norm(predicted_delta, axis=1) <= 4.0).tolist())
        if errors: macro_error.append(float(np.mean(errors))); macro_within.append(float(np.mean(within)))
    return {"identity_macro_relative_vector_error_m": float(np.mean(macro_error)), "identity_macro_within_4m_fraction": float(np.mean(macro_within)), "multi_traversal_identities": len(macro_error)}


def _direct_loss(predicted, target, event):
    import torch
    from torch.nn import functional as F
    values = []
    for code in (1, 2):
        mask = event == code
        if not bool(mask.any()): raise ValueError("vector direct batch lacks one event")
        values.append(F.smooth_l1_loss(predicted[mask], target[mask], beta=1.0))
    return torch.stack(values).mean()


def _relative_loss(predicted_center, target_center, pairs):
    import torch
    from torch.nn import functional as F
    predicted_delta = predicted_center[pairs[:, 0]] - predicted_center[pairs[:, 1]]
    target_delta = target_center[pairs[:, 0]] - target_center[pairs[:, 1]]
    return F.smooth_l1_loss(predicted_delta, target_delta, beta=1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-cache", required=True, type=Path); parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path); parser.add_argument("--scalar-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--identities-per-event", type=int, default=16); parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("formal vector-center training requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; device = torch.device("cuda")
    with np.load(args.teacher.resolve(), allow_pickle=False) as z:
        partition=z["partition_code"].astype(np.uint8); valid=z["valid_mask"].astype(bool); identity=z["identity"].astype(str); event_name=z["event"].astype(str)
        tangent=z["route_tangent_xyz"].astype(np.float32); objective=z["objective_center_xyz_m"].astype(np.float32); longitudinal=z["signed_center_offset_m"].astype(np.float32); oracle=z["oracle_longitudinal_center_xyz_m"].astype(np.float32); global_index=z["global_sequence_index"]
    traversal=np.load(args.action_cache/"traversal_id.npy").astype(str); cache_global=np.load(args.action_cache/"global_sequence_index.npy")
    if not np.array_equal(global_index,cache_global): raise RuntimeError("vector-center cache identity drift")
    sensor=oracle-longitudinal[:,None]*tangent; local=local_event_center_vectors(sensor,objective,tangent,valid); basis=local["route_local_basis"]; target=local["local_center_vector_m"]
    event=np.where(event_name=="junction",1,np.where(event_name=="terminal",2,0)).astype(np.int64); fit=np.flatnonzero(valid&(partition==0)); selection=np.flatnonzero(valid&(partition==1))
    if len(fit)!=25294 or len(selection)!=8839: raise RuntimeError("vector-center split drift")
    fit_dataset=ActionSetNodeDataset(args.action_cache,fit); selection_dataset=ActionSetNodeDataset(args.action_cache,selection); groups=_groups(fit,identity,traversal,event)
    action=torch.load(args.action_checkpoint.resolve(),map_location=device,weights_only=False); scalar=torch.load(args.scalar_checkpoint.resolve(),map_location=device,weights_only=False)
    if action.get("seed")!=args.seed or scalar.get("schema_version") not in ("gse_event_center_offset_checkpoint_v1", "gse_event_center_dual_batch_corrective_checkpoint_v1") or scalar.get("seed")!=args.seed: raise RuntimeError("vector-center checkpoint drift")
    base=ActionSetNodeDetector().to(device); base.load_state_dict(action["model"],strict=True); base.eval()
    for parameter in base.parameters(): parameter.requires_grad_(False)
    head=EventCenterVectorHead().to(device); head.initialize_from_scalar_state(scalar["head"])
    for parameter in head.hidden.parameters(): parameter.requires_grad_(False)
    for parameter in head.longitudinal_output.parameters(): parameter.requires_grad_(False)
    trainable=[parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=args.learning_rate,weight_decay=1e-4); steps=int(np.ceil(len(fit)/64)); fit_event=event[fit]; by_event={code:np.flatnonzero(fit_event==code) for code in (1,2)}
    initial_vector=_infer(base,head,selection_dataset,len(selection),device); initial_center=_global_centers(initial_vector,selection,sensor,basis); initial_cross=_cross_metrics(initial_center,selection,identity,traversal,objective[selection]); initial_long_mae=float(np.mean(np.abs(initial_vector[:,0]-target[selection,0])))
    best=(initial_cross["identity_macro_relative_vector_error_m"],initial_long_mae); history=[]
    torch.save({"schema_version":"gse_event_center_vector_checkpoint_v1","seed":args.seed,"epoch":-1,"head":head.state_dict(),"selection_cross_view":initial_cross,"selection_longitudinal_mae_m":initial_long_mae},args.output_dir/"best.pt")
    for epoch in range(args.epochs):
        rng=np.random.default_rng(args.seed*1000+epoch); pools={code:rng.choice(values,size=steps*32,replace=steps*32>len(values)) for code,values in by_event.items()}; totals=[0.,0.]; head.train()
        for step in range(steps):
            pair_local,pairs=_sample_batch(groups,rng,args.identities_per_event); pair_rows=fit[pair_local]; direct_local=np.concatenate([pools[code][step*32:(step+1)*32] for code in (1,2)]); direct_rows=fit[direct_local]
            dt,dm=_collate(fit_dataset,direct_local); pt,pm=_collate(fit_dataset,pair_local)
            with torch.no_grad(): dc=base(torch.from_numpy(dt).to(device),torch.from_numpy(dm).to(device))["causal_context"]; pc=base(torch.from_numpy(pt).to(device),torch.from_numpy(pm).to(device))["causal_context"]
            dp=head(dc); pp=head(pc); dl=_direct_loss(dp,torch.from_numpy(target[direct_rows]).to(device),torch.from_numpy(event[direct_rows]).to(device))
            pb=torch.from_numpy(basis[pair_rows]).to(device); ps=torch.from_numpy(sensor[pair_rows]).to(device); tc=torch.from_numpy(objective[pair_rows]).to(device); predicted_center=ps+torch.einsum("nij,ni->nj",pb,pp); rl=_relative_loss(predicted_center,tc,torch.from_numpy(pairs).to(device)); loss=dl+rl
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,5.0); optimizer.step(); totals[0]+=float(dl.detach().cpu());totals[1]+=float(rl.detach().cpu())
        predicted=_infer(base,head,selection_dataset,len(selection),device); center=_global_centers(predicted,selection,sensor,basis); cross=_cross_metrics(center,selection,identity,traversal,objective[selection]); long_mae=float(np.mean(np.abs(predicted[:,0]-target[selection,0]))); record={"epoch":epoch,"direct_loss":totals[0]/steps,"relative_loss":totals[1]/steps,"selection_longitudinal_mae_m":long_mae,**cross}; history.append(record);print(json.dumps(record,sort_keys=True),flush=True)
        score=(cross["identity_macro_relative_vector_error_m"],long_mae)
        if long_mae<=initial_long_mae and score<best:
            best=score;torch.save({"schema_version":"gse_event_center_vector_checkpoint_v1","seed":args.seed,"epoch":epoch,"head":head.state_dict(),"selection_cross_view":cross,"selection_longitudinal_mae_m":long_mae},args.output_dir/"best.pt")
    checkpoint=torch.load(args.output_dir/"best.pt",map_location=device,weights_only=False);head.load_state_dict(checkpoint["head"]);predicted=_infer(base,head,selection_dataset,len(selection),device);center=_global_centers(predicted,selection,sensor,basis);cross=_cross_metrics(center,selection,identity,traversal,objective[selection])
    component_mae=np.mean(np.abs(predicted-target[selection]),axis=0);summary={"schema_version":"gse_event_center_vector_seed_v1","seed":args.seed,"best_epoch":int(checkpoint["epoch"]),"optimizer_steps":args.epochs*steps,"backbone_optimizer_steps":0,"selection_component_mae_m":component_mae.tolist(),"selection_cross_view":cross,"duration_seconds":time.monotonic()-started,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0}
    np.savez_compressed(args.output_dir/"selection_outputs.npz",observation_row=selection,global_sequence_index=global_index[selection],predicted_local_vector_m=predicted.astype(np.float32),predicted_center_xyz_m=center.astype(np.float32),target_local_vector_m=target[selection],target_center_xyz_m=objective[selection])
    (args.output_dir/"history.json").write_text(json.dumps(history,indent=2,sort_keys=True)+"\n");(args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps(summary,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
