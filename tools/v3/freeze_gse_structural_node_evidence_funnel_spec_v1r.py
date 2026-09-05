#!/usr/bin/env python3
"""Freeze the support/identity-separation corrective node-evidence audit."""

from __future__ import annotations

import copy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


SOURCE_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structural_node_evidence_funnel_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structural_node_evidence_funnel_v1r.json"
SLUG = "gse_structural_node_evidence_funnel_v1r"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("structural-node evidence corrective spec exists; overwrite is forbidden")
    spec = copy.deepcopy(load_json(SOURCE_SPEC))
    spec["slug"] = SLUG
    spec["corrective_of"] = "results/gate3_semantics/gate3_20260904_gse_structural_node_evidence_funnel_v1_seed0"
    spec["question"] = "After separating viewpoint-dependent ray support from place identity, does causal five-frame primitive composition support safe structural-node association?"
    spec["method"] = "Repeat the frozen C01-C07 audit with the sole scientific correction that support-ray count affects uncertainty only, never identity distance. Port shape, gravity-relative slope, curvature and pairwise angular composition remain the identity evidence; data, pair population, 0.98/0.25 gates and C07 isolation are unchanged."
    spec["fallback"] = "If the corrected five-frame descriptor still fails, stop direct local-descriptor association and test graph-consistent candidate restriction before any neural training."
    spec["corrective_scope"] = {
        "evidence": "V1 oracle C07 P/R=1/1 and single-frame P/R=0.981722/0.931567, while five-frame accepted 0 C07 pairs because ray support was encoded as identity.",
        "allowed_change": "Replace the identity support feature with gravity-relative port slope and exclude the final support uncertainty scalar from pair distance.",
        "unchanged": "All sources, observations, fit/C07 split, candidate pairs, thresholds, gates and resource limits."
    }
    tools = {
        "descriptor": "src/mtare_topo/representation/gse_structural_node_evidence.py",
        "descriptor_tests": "tests/v3/unit/test_gse_structural_node_evidence.py",
        "executor": "tools/v3/execute_gse_structural_node_evidence_funnel_v1.py",
        "runner": "tools/v3/run_gse_structural_node_evidence_funnel_v1.py",
        "freezer": "tools/v3/freeze_gse_structural_node_evidence_funnel_spec_v1r.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py"
    }
    spec["frozen_tools"] = {
        name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)}
        for name, relative in tools.items()
    }
    run_id = f"gate3_{spec['date']}_{SLUG}_seed{spec['seed']}"
    spec["command"] = [
        "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "3600s", PYTHON,
        "tools/v3/run_gse_structural_node_evidence_funnel_v1.py", "--spec", str(SPEC),
        "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{run_id}")
    ]
    write_json(SPEC, spec)
    print({"spec": str(SPEC), "frozen_inputs": len(spec["frozen_inputs"]), "frozen_tools": len(tools)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
