#!/usr/bin/env python3
"""Freeze the host-RSS evidence corrective for sparse-port training."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_sparse_port_three_seed_training_v1.json"
OLD_SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_sparse_port_three_seed_training_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_sparse_port_three_seed_training_v1r.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_sparse_port_three_seed_training_v1r.json"
RUN_ID = "gate3_20260901_primitive_relation_sparse_port_three_seed_training_v1r_seed0"
FAILED = PROJECT_ROOT / "results/gate3_semantics/gate3_20260831_primitive_relation_sparse_port_three_seed_training_v1_seed0"
PYTHON = (
    "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/"
    "phase3_torch290_cu129_zarr2187_v1/bin/python"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    failed_state = json.loads((FAILED / "RUN_STATE.json").read_text())
    failed_summary = json.loads((FAILED / "metrics/summary.json").read_text())
    if (
        failed_state.get("state") != "FAILED"
        or failed_summary.get("failure_class")
        != "SYSTEM_RESOURCE_EVIDENCE_CONTRACT"
        or failed_summary.get("completed_epochs") != 0
        or failed_summary.get("checkpoint_files") != 0
    ):
        raise RuntimeError("V1R resource corrective trigger is not exact")
    old_card = json.loads(OLD_CARD.read_text(encoding="utf-8"))
    old_spec = json.loads(OLD_SPEC.read_text(encoding="utf-8"))
    approval = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-01T00:05:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": [
            "training", "checkpoint_selection", "threshold_calibration",
        ],
        "confirmation_reference": (
            "User authorized continuous autonomous in-plan execution. This "
            "corrective changes only resource evidence after an operator stop; "
            "the scientific experiment remains byte-identical."
        ),
        "scope": (
            "One immutable V1R rerun with the exact V1 data/model/loss/schedule/"
            "seeds/gates plus an independent host-RSS monitor. C08 remains "
            "conditional; no C09/C10/graph/M-TARE."
        ),
    }
    card = copy.deepcopy(old_card)
    card.update({
        "card_id": "primitive_relation_sparse_port_three_seed_training_v1r",
        "status": (
            "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_"
            "SPARSE_PORT_THREE_SEED_TRAINING_V1R"
        ),
        "approval": approval,
        "purpose": (
            "Repeat the uncompleted V1 scientific contract after fixing only its "
            "host-memory evidence: independently record Linux VmHWM/RUSAGE_CHILDREN, "
            "while retaining the old nvidia-smi value explicitly as GPU-process memory."
        ),
        "failure_policy": (
            "Any source/tool/environment drift, nonfinite gradient, data mismatch, "
            "host RSS/GPU/result/time overrun, C07 failure, C08 regression or "
            "forbidden-world read stops. Never reuse partial V1 optimizer state; "
            "do not change model, data, schedule, thresholds or gates."
        ),
        "retention": (
            "Keep 21 epoch checkpoints/histories, three selected checkpoints, "
            "per-seed host/GPU resource evidence, C07 metrics, conditional C08 "
            "outputs, paper figures, logs, RUN_STATE and seal. Keep the failed V1 "
            "as operator/system evidence; do not combine its partial updates."
        ),
    })
    card["metrics_and_pre_registered_gates"]["implementation"] = (
        "Exactly 48 frozen tests pass, including live /proc VmHWM parsing, "
        "current-process observation and forced over-limit termination. V2 model "
        "remains 2,629,870 parameters and unchanged from sealed readiness."
    )
    card["metrics_and_pre_registered_gates"]["resources"] = (
        "Full run <=40 h; exact PyTorch CUDA allocation <=16 GiB; sampled GPU "
        "process memory <=16 GiB; independent Linux child peak host RSS <=16 GiB; "
        "output <=6 GiB. Zero C09/C10/graph/M-TARE."
    )
    card["metrics_and_pre_registered_gates"]["corrective_identity"] = (
        "Relative to failed V1, only the outer resource monitor and its tests/run "
        "identity change. Trainer, evaluator, data, initialization seeds, seven "
        "epochs, optimizer, losses, checkpoint selection, thresholds and science "
        "gates are identical. Failed V1 completed zero epochs and wrote zero checkpoints."
    )
    card["estimated_cost"] = {
        "compute": (
            "One RTX 5090, three seeds sequentially; hard 40 h outer limit. The "
            "discarded V1 consumed under seven minutes and no checkpoint is reused."
        ),
        "wall_time_hours": 36,
        "host_ram_gb": 16,
        "gpu": 1,
        "gpu_memory_gb": 16,
        "disk_gb": 6,
    }
    write(CARD, card)

    tool_paths = {
        name: record["path"] for name, record in old_spec["frozen_tools"].items()
    }
    tool_paths["v1_runner_dependency"] = tool_paths["runner"]
    tool_paths["runner"] = (
        "tools/v3/run_primitive_relation_sparse_port_three_seed_training_v1r.py"
    )
    tool_paths["resource_monitor_tests"] = (
        "tests/v3/unit/test_primitive_relation_sparse_port_resource_monitor_v1r.py"
    )
    input_paths = [
        PROJECT_ROOT / relative
        for relative in old_spec["frozen_inputs"]
        if relative != str(OLD_CARD.relative_to(PROJECT_ROOT))
    ]
    input_paths.extend([
        CARD,
        FAILED / "RUN_STATE.json",
        FAILED / "metrics/summary.json",
        FAILED / "artifacts/evidence_sha256.txt",
    ])
    expected_counts = copy.deepcopy(old_spec["expected_counts"])
    expected_counts["unit_tests"] = 48
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "operation": "training",
        "date": "20260901",
        "slug": "primitive_relation_sparse_port_three_seed_training_v1r",
        "seed": 0,
        "question": old_spec["question"],
        "method": (
            old_spec["method"]
            + " V1R adds independent Linux host peak-RSS evidence only."
        ),
        "baseline": old_spec["baseline"],
        "fallback": old_spec["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": expected_counts,
        "expected_evidence": [
            "Three seven-epoch histories/checkpoints; per-seed independent host "
            "VmHWM and explicit GPU-process/PyTorch peaks; complete C07 geometry, "
            "relation, uncertainty and non-vacuous safety comparisons; conditional "
            "C08 outputs; raw logs, paper figure, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_tools": {
            name: {"path": path, "sha256": sha(PROJECT_ROOT / path)}
            for name, path in tool_paths.items()
        },
        "frozen_inputs": {
            str(path.relative_to(PROJECT_ROOT)): sha(path) for path in input_paths
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=GSE sparse-port V1R resource-corrected three-seed training",
            "--mode=block", "/usr/bin/timeout", "--signal=INT",
            "--kill-after=60s", "144000s", "/usr/bin/env",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}",
            PYTHON,
            "tools/v3/run_primitive_relation_sparse_port_three_seed_training_v1r.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
