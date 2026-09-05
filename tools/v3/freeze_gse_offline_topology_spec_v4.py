#!/usr/bin/env python3
"""Freeze the one-time C09 node-gated topology corrective spec."""

from __future__ import annotations

import copy
import hashlib
import json

from _bootstrap import PROJECT_ROOT
import freeze_gse_offline_topology_spec_v3 as v3


RUN_ID = "gate4_20260826_gse_offline_topology_validation_v4_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate4/gse_offline_topology_validation_v4.json"
CARD = "configs/v3/gate4/data_cards/gse_offline_topology_validation_v4.json"
NODE_RUN = "results/gate4_topology/gate4_20260826_gse_node_matchability_ensemble_calibration_v1_seed0"
TOOLS = {
    **v3.TOOLS,
    "data_card": CARD,
    "evaluator": "tools/v3/evaluate_gse_offline_topology_v4.py",
    "v3_evaluator_core": "tools/v3/evaluate_gse_offline_topology_v3.py",
    "runner": "tools/v3/run_gse_offline_topology_validation_v4.py",
    "v3_runner_core": "tools/v3/run_gse_offline_topology_validation_v3.py",
    "ensemble_runtime": "src/mtare_topo/representation/gse_exit_token_ensemble.py",
}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("C09 V4 spec already exists")
    source = json.loads((PROJECT_ROOT / "configs/v3/gate4/gse_offline_topology_validation_v3.json").read_text(encoding="utf-8"))
    spec = copy.deepcopy(source)
    run_dir = PROJECT_ROOT / "results/gate4_topology" / RUN_ID
    command = [
        "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
        "--why=GSE C09 node-gated topology corrective", "--mode=block",
        "/usr/bin/timeout", "--signal=INT", "--kill-after=300s", "29400s",
        "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
        "tools/v3/run_gse_offline_topology_validation_v4.py",
        "--spec", str(SPEC), "--run-dir", str(run_dir),
    ]
    spec.update({
        "slug": "gse_offline_topology_validation_v4",
        "question": "Does the C01-C08-frozen open-set node gate make the complete GSE graph association-safe and at least five F1 points better than the strongest deployable C09 baseline?",
        "method": "Run all baselines first. The full method applies the frozen equal-weight matchability gate at 0.982292910416921 before non-corridor node triggers, then the frozen pair ensemble at 0.9431912302970886 inside 16 m; graph grid, Teacher, metrics and traversal-only edge policy are unchanged.",
        "data_card": CARD,
        "config_path": CARD,
        "command": command,
        "estimated_cost": {
            "compute": "Three frozen node scores per C09 observation, three frozen pair scores per unique <=16m candidate, and the unchanged 59,442,660 typed graph updates; zero model updates.",
            "wall_time_hours": 8.0, "disk_gb": 5.0,
        },
    })
    spec["acceptance_criteria"] = [
        "All three baselines complete exact 243-row sweeps before the full method; all ten worlds, 2054 traversals and 24462 sequences remain unchanged.",
        "Full GSE has nonzero merges, precision>=0.98 and false-loop<=0.01 without changing frozen node/pair thresholds or 16m domain.",
        "Full GSE node F1 and edge F1 each exceed the strongest deployable baseline by at least 0.05.",
        "Absolute mean signed connected-component and cycle-rank errors are each <=0.25; zero C10/M-TARE/model updates.",
    ]
    node_inputs = [
        f"{NODE_RUN}/RUN_STATE.json",
        f"{NODE_RUN}/metrics/summary.json",
        f"{NODE_RUN}/artifacts/evidence_sha256.txt",
        f"{NODE_RUN}/artifacts/calibration/summary.json",
    ]
    all_inputs = list(source["frozen_inputs"]) + node_inputs
    spec["frozen_inputs"] = {relative: _sha(PROJECT_ROOT / relative) for relative in all_inputs}
    spec["frozen_tools"] = {
        name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)}
        for name, relative in TOOLS.items()
    }
    spec["user_authorization"] = {
        "status": "APPROVED", "approved_by": "user",
        "approved_at": "2026-08-26T17:25:00+08:00",
        "scope": "One immutable baselines-first C09 V4 topology corrective using only the separately frozen node gate added to V3.",
        "confirmation_reference": "Standing authorization for autonomous GSE-Graph paper execution without repeated approval prompts.",
    }
    SPEC.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
