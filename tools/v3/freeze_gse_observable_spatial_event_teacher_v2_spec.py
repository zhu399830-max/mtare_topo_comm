#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate2/gse_observable_spatial_event_teacher_v2.json"
DATA_CARD = "configs/v3/gate2/data_cards/gse_observable_spatial_event_teacher_v2.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("observable Teacher V2 spec already exists")
    source = "results/gate2_representation/gate2_20260828_gse_spatial_multi_event_teacher_export_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{source}/RUN_STATE.json",
        f"{source}/metrics/summary.json",
        f"{source}/artifacts/evidence_sha256.txt",
        f"{source}/artifacts/export/summary.json",
        f"{source}/artifacts/export/event_identity_map.json",
        f"{source}/artifacts/export/teacher_shard_manifest.jsonl",
    ]
    tools = {
        "filter": "src/mtare_topo/teacher/gse_observable_spatial_event_teacher.py",
        "executor": "tools/v3/execute_gse_observable_spatial_event_teacher_v2.py",
        "runner": "tools/v3/run_gse_observable_spatial_event_teacher_v2.py",
        "freezer": "tools/v3/freeze_gse_observable_spatial_event_teacher_v2_spec.py",
        "tests": "tests/v3/unit/test_gse_observable_spatial_event_teacher.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    authorization = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T21:00:00+08:00",
        "authorized_gates": [2],
        "authorized_operations": ["teacher_generation"],
        "scope": "One immutable C01-C08 observable spatial-event Teacher V2 export using only the frozen LiDAR vertical FOV.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 2,
        "execution_phase": 3,
        "date": "20260828",
        "slug": "gse_observable_spatial_event_teacher_v2",
        "seed": 0,
        "operation": "teacher_generation",
        "data_card": DATA_CARD,
        "question": "Can the sealed spatial event Teacher be corrected to the fixed 16-line LiDAR vertical FOV without deleting observations, losing identities, reading model outputs or touching test worlds?",
        "method": "Intersect every active V1 token with the fixed inclusive target-elevation interval [-15,+15] degrees, repack retained tokens in original order, preserve all rows and identity indices, and record every removed token with provenance.",
        "baseline": "Teacher V1 uses native-mesh continuous-ray LOS but omits the discrete sensor vertical FOV, causing the geometry-anchored readiness support failure.",
        "fallback": "If exact predeclared counts, all identities or row preservation fail, stop and audit only the Teacher coordinate/FOV contract; do not change sensor or inspect model errors.",
        "user_authorization": authorization,
        "acceptance_criteria": [
            "Exact 80 worlds and all 188126 observations retained with unchanged global sequence identities.",
            "Remove exactly 1631 outside-FOV tokens and retain exactly 131424 tokens: terminal 30714, junction 100710.",
            "Retain all 1076 identities; fit/selection tokens 98279/33145; cardinality 0-5 exactly 82326/82183/21756/1724/128/9.",
            "Filter consumes only sealed relative xyz/mask and fixed FOV, never prediction/error/checkpoint/test information.",
            "Zero training, inference, normalization, threshold selection, C09/C10/M-TARE; sources unchanged and evidence sealed.",
        ],
        "expected_counts": {
            "worlds": 80,
            "observations": 188126,
            "source_tokens": 133055,
            "observable_tokens": 131424,
            "removed_tokens": 1631,
            "event_identities": 1076,
            "optimizer_steps": 0,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "80 immutable V2 Zarr shards, identity map and shard manifest.",
            "1631-row removed-token provenance table and exact population summary.",
            "Paper PNG/PDF/SVG and figure source, full log, environment, RUN_STATE and SHA-256 seal.",
        ],
        "estimated_cost": {
            "compute": "CPU-only transform of 80 small sealed Teacher shards; no sensor/model/mesh reads.",
            "wall_time_hours": 0.05,
            "host_ram_gb": 2,
            "gpu_memory_gb": 0,
            "disk_gb": 0.1,
            "gpu": "none",
        },
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {
            name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)}
            for name, path in tools.items()
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout",
            "900s",
            PYTHON,
            "tools/v3/run_gse_observable_spatial_event_teacher_v2.py",
            "--spec",
            str(SPEC),
            "--run-dir",
            str(PROJECT_ROOT / f"results/gate2_representation/{RUN_ID}"),
        ],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
