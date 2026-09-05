#!/usr/bin/env python3
"""Read frozen final scores; no model execution, fitting or calibration."""
import argparse
import html
import json
import os
from pathlib import Path
import platform
import resource
import signal
import time
import traceback

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_geometry_bound_rank_inputs import load_rank_inputs
from mtare_topo.evaluation.gse_binary_ranking import binary_ranking
from mtare_topo.evaluation.gse_partial_structure import evaluate_partial_structure
from mtare_topo.governance import build_run_id, load_json, write_json
from mtare_topo.governance_geometry_bound_rank import validate_geometry_bound_rank_card
from mtare_topo.representation.gse_partial_structure_training import BRANCHES
from run_gse_assignment_attribution_v1 import sha, sliced


def contained(relative):
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("path outside project")
    return path


def population(prediction, target, scored):
    """Only frozen independent assignments, never label-based selection."""
    qmap = scored.unique_center_query
    row, tid, mid = torch.where(scored.scored_member_mask)
    query = qmap[row, tid]
    logits = prediction.membership_logits[row, query, mid]
    labels = target.members[row, tid, mid]
    erow, etid = torch.where(target.event_valid & (qmap >= 0))
    eq = qmap[erow, etid]
    event_logits = prediction.event_logits[erow, eq]
    # Stable one-vs-rest junction logit; no new binary decision is used.
    junction_logit = event_logits[:, 1].double() - torch.logsumexp(event_logits[:, (0, 2)].double(), -1)
    return {
        "members": {"observation": row, "target": tid, "query": query, "member": mid,
            "logit": logits, "probability": torch.sigmoid(logits), "label": labels},
        "junction": {"observation": erow, "target": etid, "query": eq,
            "logit": junction_logit, "probability": torch.softmax(event_logits, -1)[:, 1],
            "label": (target.events[erow, etid] == 1).to(torch.float64),
            "original_argmax": event_logits.argmax(-1), "original_event": target.events[erow, etid]},
    }


def describe(values, selected):
    score = values["probability"][selected].cpu().numpy()
    label = values["label"][selected].cpu().numpy()
    result = binary_ranking(score, label)
    if len(label):
        result["final_logit_bce"] = float(F.binary_cross_entropy_with_logits(
            values["logit"][selected].double(), values["label"][selected].double()))
        p = float(np.mean(label))
        result["constant_prior_bce"] = float(-sum(x * np.log(x) for x in (p, 1-p) if x > 0))
    else:
        result.update(final_logit_bce=None, constant_prior_bce=None)
    return result


def preview(task, populations, indices):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="820">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="15" y="25">C01 FIT / {html.escape(task)} / all selected observations</text>',
        '<text x="15" y="48">Fixed final score distributions: blue positive, orange negative; each class normalized separately.</text>',
        '<text x="15" y="70">No threshold selection. Member decision remains 0.5; events remain original 3-class argmax.</text>']
    for br, branch in enumerate(BRANCHES):
        for col, kind in enumerate(("members", "junction")):
            v = populations[branch][kind]
            choose = np.isin(v["observation"].numpy(), indices)
            probs, labels = v["probability"].numpy()[choose], v["label"].numpy()[choose]
            x, y = 35 + col*510, 100 + br*235
            parts.extend([f'<text x="{x}" y="{y}">{html.escape(branch)} / {kind}</text>',
                f'<rect x="{x}" y="{y+20}" width="450" height="150" fill="none" stroke="#bbb"/>'])
            for positive, color, offset in ((True, "#2166ac", 0), (False, "#d95f02", 10)):
                selected_probs = probs[labels == int(positive)]
                hist, _ = np.histogram(selected_probs, bins=np.linspace(0, 1, 21))
                normalized = hist / max(1, len(selected_probs))
                for i, frequency in enumerate(normalized):
                    h = float(frequency)*150
                    parts.append(f'<rect x="{x+i*22.5+offset:.3f}" y="{y+170-h:.3f}" width="10" height="{h:.3f}" fill="{color}"/>')
            parts.extend([f'<text x="{x}" y="{y+190}">0</text>',
                f'<text x="{x+220}" y="{y+190}">0.5</text>', f'<text x="{x+430}" y="{y+190}">1.0</text>',
                f'<text x="{x}" y="{y+212}">positive={int((labels==1).sum())}; negative={int((labels==0).sum())}</text>'])
    return "\n".join(parts + ["</svg>"])


def analyze_and_save(run, loaded):
    data, output, populations = loaded["data"], {}, {}
    manifest = data["manifest"]
    tasks = sorted({r["task"] for r in manifest})
    for branch in BRANCHES:
        key = "gt" if branch == "gt_axes" else "predicted"
        target, prediction = data["targets"][key], loaded["predictions"][branch]
        bridge = [r["target_transport"] for r in data["transport"][key]]
        scored = evaluate_partial_structure(prediction, target, membership_threshold=.5,
            manifest=manifest, direction_bridge_ledger=bridge)
        old = loaded["training_summary"]["result"]["evaluation"]["final"][branch]
        if scored.summary != old["aggregate"]:
            raise ValueError("original aggregate drift: " + branch)
        for name in ("unique_center_query", "scored_member_mask"):
            if not torch.equal(getattr(scored, name), loaded["saved_scoring"][branch][name]):
                raise ValueError("original saved scoring drift: " + name)
        populations[branch] = population(prediction, target, scored)
        groups = {"all": list(range(len(manifest))), **{task: [i for i,r in enumerate(manifest) if r["task"] == task] for task in tasks}}
        if set(old["parents"]) != set(tasks):
            raise ValueError("parent population drift")
        ranks = {}
        for group, indices in groups.items():
            if group != "all":
                check = evaluate_partial_structure(sliced(prediction, indices), sliced(target, indices),
                    membership_threshold=.5, manifest=[manifest[i] for i in indices],
                    direction_bridge_ledger=[bridge[i] for i in indices]).summary
                if check != old["parents"][group]:
                    raise ValueError("original parent score drift: " + group)
            ranks[group] = {kind: describe(v, torch.from_numpy(np.isin(v["observation"].numpy(), indices)))
                for kind,v in populations[branch].items()}
            # The junction ranking is not a new binary event classifier.
            for name in ("confusion", "fixed_threshold"):
                del ranks[group]["junction"][name]
            ranks[group]["junction"]["decision"] = "original_three_class_argmax_unchanged"
        expected_confusion = {k: old["aggregate"]["members"][k] for k in ("tp", "fp", "fn", "tn")}
        if ranks["all"]["members"]["confusion"] != expected_confusion:
            raise ValueError("fixed member decision drift")
        write_json(run / "artifacts" / (branch + "__scored_probabilities.json"),
            {kind: {k: v.tolist() for k,v in values.items()} for kind,values in populations[branch].items()})
        output[branch] = ranks
        record = {"branch": branch, "original_aggregate_parents_masks_exact": True,
            "members": ranks["all"]["members"], "junction": ranks["all"]["junction"]}
        with (run / "logs/diagnostic.jsonl").open("a") as log:
            log.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
    for task in tasks:
        indices = [i for i,r in enumerate(manifest) if r["task"] == task]
        (run / "previews" / (task + ".svg")).write_text(preview(task, populations, indices))
    write_json(run / "artifacts/manifest.json", manifest)
    return {"branches": output, "original_scores_reproduced": True,
        "cached_prediction_observations": len(BRANCHES)*len(manifest), "model_inference": 0,
        "checkpoint_reads": 0, "optimizer_steps": 0, "threshold_search": False,
        "diagnostic_not_calibration": True, "parent_plots": len(tasks)}


def execute(spec, run):
    run = Path(run).resolve()
    if (run != PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise RuntimeError("requires fresh exact run; no overwrite/retry")
    started, result, error, reads = time.monotonic(), {}, None, {}
    write_json(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name})
    def deadline(signum, frame):
        raise TimeoutError("rank diagnosis exceeds 300 seconds")
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(300)
    try:
        card = load_json(contained(spec["data_card"]))
        report = validate_geometry_bound_rank_card(card)
        if (not report.passed or spec["operation"] != "data_export" or spec["wall_time_cap_s"] != 300
                or load_json(run / "config/data_card.json") != card or spec["rank_policy"] != card["rank_policy"]):
            raise ValueError("rank scope drift: " + str(report.errors))
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
            raise ValueError("CPU only: CUDA_VISIBLE_DEVICES must be empty from start")
        frozen = {**card["sealed_sources"], **spec["source_sha256"]}
        for path,digest in frozen.items():
            if sha(contained(path)) != digest:
                raise ValueError("frozen source drift: " + path)
        versions = {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "zarr": zarr.__version__, "scipy": scipy.__version__}
        if versions != spec["expected_versions"]:
            raise ValueError("environment drift")
        write_json(run / "config/execution_environment.json", {**versions,"platform":platform.platform(),"device":"cpu"})
        loaded = load_rank_inputs(PROJECT_ROOT, card)
        reads = loaded["read_hashes"]
        result = analyze_and_save(run, loaded)
        if result["cached_prediction_observations"] != 540 or len(reads) != 8:
            raise ValueError("exact eight sources and 540 cached outputs required")
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if (result["peak_rss_bytes"] > 4*1024**3 or time.monotonic()-started > 300
                or sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) > 490_000_000):
            raise RuntimeError("resource cap exceeded")
        for path,digest in {**frozen,**reads}.items():
            if sha(contained(path)) != digest:
                raise ValueError("source changed during diagnosis: " + path)
    except Exception:
        error = traceback.format_exc()
        (run / "logs/error.log").write_text(error)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    write_json(run / "artifacts/source_reads_sha256.json", reads)
    status = "GEOMETRY_BOUND_RANK_DIAGNOSTIC_COMPLETE" if error is None else "GEOMETRY_BOUND_RANK_DIAGNOSTIC_FAIL"
    write_json(run / "metrics/summary.json", {"status":status,"elapsed_s":time.monotonic()-started,
        "result":result,"error":error,"scientific_gate_pass":False})
    write_json(run / "RUN_STATE.json", {"state":"COMPLETED" if error is None else "FAILED","run_id":run.name,"error":error})
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p != seal))
    print(json.dumps({"status":status,"elapsed_s":time.monotonic()-started,"error":error,"seal_sha256":sha(seal)}),flush=True)
    return int(error is not None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    args = parser.parse_args()
    return execute(load_json(args.spec),args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
