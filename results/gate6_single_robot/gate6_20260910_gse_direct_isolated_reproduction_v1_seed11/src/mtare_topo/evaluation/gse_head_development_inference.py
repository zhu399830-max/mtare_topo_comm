"""Frozen raw-axis head development inference, never a detector or optimizer."""
import math
import numpy as np
import torch
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points
from mtare_topo.evaluation.gse_axis_observation_dependence import score_observation,summarize

GEOMETRY_FIELDS=frozenset(("frame_row","source_global_sequence_index","primitive_mask","axis_control_current_sensor_m"))


@torch.no_grad()
def infer_task(model,head,student,guard,repeat=False):
    device=next(model.parameters()).device;b=len(student)
    if b!=18:raise ValueError("exact18 observations per task required")
    values=[torch.from_numpy(np.stack([getattr(row,key) for row in student])).to(device)
            for key in ("range_valid","relative_translation_current_sensor_m","relative_yaw_current_sensor_deg")]
    guard();legacy=model(*values).axis_control_current_sensor_m.detach().cpu().numpy()
    if repeat and not np.array_equal(legacy,model(*values).axis_control_current_sensor_m.detach().cpu().numpy()):raise ValueError("legacy repeat mismatch")
    memory,valid_memory,xyz,_=model._memory(*values)
    slots=model.slot_decoder(model.slot_query[None].expand(b,-1,-1),memory,memory_key_padding_mask=~valid_memory)
    query=model.control_query(slots).reshape(b,32,3,128)
    logits=torch.einsum("bscd,bnd->bscn",query,memory)/math.sqrt(128)
    weights=torch.softmax(logits.masked_fill(~valid_memory[:,None,None],-torch.inf),dim=-1)
    reconstructed=torch.einsum("bscn,bnd->bscd",weights,xyz)
    if not np.array_equal(legacy,reconstructed.cpu().numpy()):raise ValueError("frozen memory/readout parity failed")
    points,valid=register_causal_lidar_points(*values)
    indices=(torch.arange(5,device=device)[:,None,None]*180+torch.arange(720,device=device)[None,None]//4).expand(5,16,720).reshape(1,-1)
    points=points.reshape(b,-1,3);valid=valid.reshape(b,-1)
    def one(i):
        guard()
        out=head(points[i:i+1],valid[i:i+1],memory[i:i+1],xyz[i:i+1],indices,slots[i:i+1]).votes.axis_control_m.detach().cpu().numpy()[0]
        if out.shape!=(32,3,3) or out.dtype!=np.float32 or not np.isfinite(out).all():raise ValueError("raw axis output drift")
        return out
    raw=np.stack([one(i) for i in range(b)])
    if repeat and not np.array_equal(raw,np.stack([one(i) for i in range(b)])):raise ValueError("raw repeat mismatch")
    if legacy.shape!=raw.shape or legacy.dtype!=np.float32 or not np.isfinite(legacy).all():raise ValueError("legacy output drift")
    return {"raw_no_offset":raw,"legacy_frozen":legacy}


def read_geometry(reader,task,batch,records):
    ids=np.asarray([r["row_index"] for r in records],dtype=np.int64)
    group=reader._open(reader.teacher_root,task,GEOMETRY_FIELDS)
    target=np.asarray(group["axis_control_current_sensor_m"].oindex[ids]);mask=np.asarray(group["primitive_mask"].oindex[ids])
    if (target.shape!=(18,32,3,3) or target.dtype!=np.float32 or mask.shape!=(18,32) or mask.dtype.kind not in "iu" or not np.isin(mask,(0,1)).all()
            or not mask.any(axis=1).all() or not np.isfinite(target[mask.astype(bool)]).all()
            or not np.array_equal(batch.frame_rows,[r["frame_rows"] for r in records])
            or not np.array_equal(mask.sum(axis=1),[r["visible_fragments"] for r in records])
            or not np.array_equal(np.asarray(group["frame_row"].oindex[ids]),batch.frame_rows)
            or not np.array_equal(np.asarray(group["source_global_sequence_index"].oindex[ids]),batch.source_sequence_indices)):
        raise ValueError("sealed metadata/geometry population drift")
    return target,mask.astype(bool)


def evaluate_outputs(axes,targets,masks,records):
    if set(axes)!={"raw_no_offset","legacy_frozen"} or targets.shape!=(180,32,3,3) or masks.shape!=(180,32) or len(records)!=180:
        raise ValueError("exact development outputs required")
    details={};results={}
    for method,values in axes.items():
        if values.shape!=(180,32,3,3):raise ValueError("prediction population drift")
        rows=[score_observation(values[i],torch.from_numpy(targets[i:i+1]),torch.from_numpy(masks[i:i+1])) for i in range(180)]
        details[method]=rows;results[method]=summarize(rows,records)
    raw,old=results["raw_no_offset"],results["legacy_frozen"]
    parent_deltas={p:old["parents"][p]["coordinate_mae_m"]-raw["parents"][p]["coordinate_mae_m"] for p in raw["parents"]}
    results["comparison"]={"old_minus_raw_coordinate_mae_m":old["macro"]["coordinate_mae_m"]-raw["macro"]["coordinate_mae_m"],
        "per_parent_old_minus_raw_coordinate_mae_m":parent_deltas,"parents_raw_better":sum(d>0 for d in parent_deltas.values()),
        "scientific_gate_pass":False,"whole_model_unseen_claim":False,"detection_claim":False}
    return details,results


def plot_development(run,axes,targets,masks,records,result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parents=sorted(result["raw_no_offset"]["parents"])
    fig,ax=plt.subplots(figsize=(10,4))
    for j,name in enumerate(("raw_no_offset","legacy_frozen")):
        ax.bar(np.arange(10)+(j-.5)*.35,[result[name]["parents"][p]["coordinate_mae_m"] for p in parents],.35,label=name)
    ax.set_xticks(np.arange(10),[p[:3] for p in parents]);ax.set_ylabel("Coordinate MAE (m)");ax.legend()
    ax.set_title("C02 head-development: 180 fixed quantile rows; backbone previously exposed\nGeometry matching, not detection or strict unseen-model test",fontsize=10)
    fig.tight_layout();fig.savefig(run/"previews/parent_geometry_mae.svg");plt.close(fig)
    for task in parents:
        positions=[i for i,r in enumerate(records) if r["task"]==task]
        fig,axs=plt.subplots(6,6,figsize=(18,18))
        for row,index in enumerate(positions):
            for view,dims in enumerate(((0,1),(0,2))):
                ax=axs.flat[2*row+view]
                for name,color in (("legacy_frozen","#df7726"),("raw_no_offset","#3587ba")):
                    for line in axes[name][index]:ax.plot(line[:,dims[0]],line[:,dims[1]],color=color,alpha=.3,lw=.6)
                for line in targets[index][masks[index]]:ax.plot(line[:,dims[0]],line[:,dims[1]],color="black",lw=1.)
                ax.set_title(f"row{records[index]['row_index']} {'XY' if view==0 else 'XZ'} (m)",fontsize=8)
                ax.set_aspect("equal",adjustable="datalim");ax.tick_params(labelsize=6)
        fig.suptitle(f"{task}: all18 observations/all32queries; teacher black, raw blue, old orange\nHead-development only; old backbone exposure; no query filtering",fontsize=11)
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(run/f"previews/{task}.svg");plt.close(fig)
