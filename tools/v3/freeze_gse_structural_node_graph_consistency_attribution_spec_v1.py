#!/usr/bin/env python3
"""Freeze one graph-consistency attribution for the 53 node false pairs."""

from __future__ import annotations

import copy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


SOURCE_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structural_node_evidence_funnel_v1r.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structural_node_graph_consistency_attribution_v1.json"
SLUG = "gse_structural_node_graph_consistency_attribution_v1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("graph-consistency attribution spec exists; overwrite is forbidden")
    spec = copy.deepcopy(load_json(SOURCE_SPEC))
    spec["slug"] = SLUG
    spec["question"] = "Can the historically frozen 16 m graph-position candidate gate remove all 53 C07 descriptor false matches without recall loss, including 1 m bounded error per node?"
    spec["method"] = "Reproduce the V1R contract failure from its sealed summary, emit all 53 false-pair identities for attribution, then intersect the unchanged node descriptor proposal with the pre-existing 16 m graph-position candidate radius. Also evaluate adversarial independent position error bounded by 1 m per node."
    spec["baseline"] = "Descriptor-only V1R on C07: 2960 TP, 53 FP, precision 0.982410, recall 0.933459, accepted-pair false fraction 0.017590."
    spec["fallback"] = "If the fixed graph gate fails, do not train; test causal neighbor/route consistency as a separate zero-training candidate mechanism."
    spec["corrective_of"] = "results/gate3_semantics/gate3_20260904_gse_structural_node_evidence_funnel_v1r_seed0"
    spec["acceptance_criteria"] = [
        "Explicitly rescore the source as FORMAL_MACHINE_PASS_RESEARCH_CONTRACT_FAIL.",
        "Record raw fit observations 36318, effective fit observations 36312 and exactly six no-window traversals; C07 effective observations 5874.",
        "At the unchanged descriptor threshold and fixed 16 m gate, C07 precision>=0.99, accepted-pair false fraction<=0.01, recall>=0.25 and all ten families have TP.",
        "Under worst-case 1 m position error per node, zero C07 false accepts and no recall loss.",
        "Zero optimizer/model inference/C08-C10/graph replay/M-TARE."
    ]
    spec["expected_evidence"] = ["Source contract rescore, all 53 false-pair records with spatial separation, fit/C07 descriptor-versus-graph metrics, bounded-error audit, per-family metrics, source integrity and seal."]
    spec["estimated_cost"] = {"compute": "CPU-only pair rescore", "wall_time_hours": 0.5, "host_ram_gb": 4, "disk_gb": 0.1, "gpu": 0}
    spec["expected_counts"] = {"fit_tasks": 180, "c07_tasks": 30, "raw_fit_observations": 36318, "effective_fit_observations": 36312, "no_window_fit_observations": 6, "effective_c07_observations": 5874, "source_c07_false_pairs": 53, "optimizer_steps": 0, "model_inference_frames": 0, "c08_c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_runs": 0}
    source = "results/gate3_semantics/gate3_20260904_gse_structural_node_evidence_funnel_v1r_seed0"
    for relative in (f"{source}/RUN_STATE.json", f"{source}/metrics/funnel/summary.json", f"{source}/artifacts/evidence_sha256.txt"):
        spec["frozen_inputs"][relative] = _sha(PROJECT_ROOT / relative)
    tools = {
        "descriptor": "src/mtare_topo/representation/gse_structural_node_evidence.py",
        "source_executor": "tools/v3/execute_gse_structural_node_evidence_funnel_v1.py",
        "executor": "tools/v3/execute_gse_structural_node_graph_consistency_attribution_v1.py",
        "runner": "tools/v3/run_gse_structural_node_graph_consistency_attribution_v1.py",
        "freezer": "tools/v3/freeze_gse_structural_node_graph_consistency_attribution_spec_v1.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py"
    }
    spec["frozen_tools"] = {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()}
    run_id = f"gate3_{spec['date']}_{SLUG}_seed{spec['seed']}"
    spec["command"] = ["/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "3600s", PYTHON, "tools/v3/run_gse_structural_node_graph_consistency_attribution_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{run_id}")]
    write_json(SPEC, spec)
    print({"spec": str(SPEC), "frozen_inputs": len(spec["frozen_inputs"]), "frozen_tools": len(tools)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
