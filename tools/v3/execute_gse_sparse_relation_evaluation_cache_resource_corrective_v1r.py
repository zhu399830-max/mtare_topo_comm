#!/usr/bin/env python3
"""Qualify per-world cache clearing against sealed batch256 evidence."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mtare_topo.governance import load_json, write_json

PASS = "PASS_GSE_SPARSE_RELATION_EVALUATION_CACHE_RESOURCE_CORRECTIVE_V1R"; FAIL = "FAIL_GSE_SPARSE_RELATION_EVALUATION_CACHE_RESOURCE_CORRECTIVE_V1R"; LIMIT = 16 * 1024**3; TOLERANCE = 3e-6

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--probe",required=True,type=Path); parser.add_argument("--baseline",required=True,type=Path); parser.add_argument("--output-dir",required=True,type=Path); args=parser.parse_args(); started=time.monotonic(); output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    probe=load_json(args.probe.resolve()/"summary.json"); baseline=load_json(args.baseline.resolve()/"summary.json"); current=dict(np.load(args.probe.resolve()/"first256_outputs.npz")); reference=dict(np.load(args.baseline.resolve()/"outputs.npz"))
    if set(current)!=set(reference): raise RuntimeError("cache parity output key drift")
    errors={name:(0.0 if np.issubdtype(current[name].dtype,np.integer) and np.array_equal(current[name],reference[name]) else float(np.max(np.abs(current[name].astype(np.float64)-reference[name].astype(np.float64))))) for name in sorted(current)}
    loss_error={name:abs(probe["first256_loss"][name]-baseline["loss"][name]) for name in baseline["loss"]}
    per_world=probe["per_world"]; memory_keys=("peak_cuda_allocated_bytes","peak_cuda_reserved_bytes","nvidia_process_memory_bytes"); maximum={key:max(row[key] for row in per_world) for key in memory_keys}
    checks={"exact_population":probe["fit_backward_rows"]==128 and probe["c07_worlds"]==10 and probe["c07_rows"]==21548,"batch256_unchanged":probe["evaluation_batch_size"]==baseline["evaluation_batch_size"]==256,"integer_outputs_exact":all(errors[name]==0 for name in errors if np.issubdtype(current[name].dtype,np.integer)),"floating_outputs_equivalent":max(errors.values())<=TOLERANCE,"first256_losses_equivalent":max(loss_error.values())<=TOLERANCE,"all_world_allocated_below_16gib":maximum["peak_cuda_allocated_bytes"]<=LIMIT,"all_world_reserved_below_16gib":maximum["peak_cuda_reserved_bytes"]<=LIMIT,"all_world_process_memory_below_16gib":maximum["nvidia_process_memory_bytes"]<=LIMIT,"cache_cleared_at_every_world_boundary":probe["cache_clear_calls"]>=2*probe["c07_worlds"],"zero_optimizer_checkpoint_forbidden_reads":probe["optimizer_steps"]==0 and probe["checkpoints_written"]==0 and all(probe[key]==0 for key in ("c08_worlds_read","c09_worlds_read","c10_worlds_read","mtare_worlds_read","graph_replays","planner_calls"))}; checks={k:bool(v) for k,v in checks.items()}; passed=all(checks.values())
    summary={"schema_version":"gse_sparse_relation_evaluation_cache_resource_corrective_v1r","status":PASS if passed else FAIL,"scientific_pass":passed,"decision":"ALLOW_PER_WORLD_CACHE_CLEAR_BATCH256_TRAINING" if passed else "STOP_BEFORE_TRAINING","memory_limit_bytes":LIMIT,"maximum_memory":maximum,"per_world":per_world,"per_output_max_abs_error":errors,"per_loss_abs_error":loss_error,"authoritative_max_output_error":max(errors.values()),"authoritative_max_loss_error":max(loss_error.values()),"checks":checks,"duration_seconds":time.monotonic()-started,"optimizer_steps":0,"checkpoints_written":0,"c08_worlds_read":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0,"planner_calls":0}
    write_json(output/"summary.json",summary); write_json(output/"figure_source.json",summary)
    x=np.arange(len(per_world)); fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for key,label in zip(memory_keys,("allocated","reserved","nvidia process")): axes[0].plot(x,[row[key]/1024**3 for row in per_world],marker="o",label=label)
    axes[0].axhline(16,color="red",ls="--"); axes[0].set(xticks=x,xticklabels=[row["parent_id"].split("_")[0] for row in per_world],ylabel="GiB",title="Full C07 batch256 memory by world"); axes[0].legend(); axes[0].grid(alpha=.2)
    top=sorted(errors.items(),key=lambda item:item[1],reverse=True)[:8]; axes[1].barh([n for n,_ in reversed(top)],[v for _,v in reversed(top)]); axes[1].axvline(TOLERANCE,color="red",ls="--"); axes[1].set(title="Cache clear output parity",xlabel="maximum absolute error"); axes[1].grid(axis="x",alpha=.2)
    fig.suptitle("Sparse relation per-world CUDA cache qualification")
    for suffix in ("png","pdf","svg"): fig.savefig(output/f"gse_sparse_relation_evaluation_cache_resource_corrective_v1r.{suffix}",dpi=220)
    plt.close(fig); print(json.dumps({"status":summary["status"],"decision":summary["decision"],"maximum_memory":maximum,"max_output_error":summary["authoritative_max_output_error"],"max_loss_error":summary["authoritative_max_loss_error"],"checks":checks},indent=2,sort_keys=True)); return 0 if passed else 2
if __name__=="__main__": raise SystemExit(main())
