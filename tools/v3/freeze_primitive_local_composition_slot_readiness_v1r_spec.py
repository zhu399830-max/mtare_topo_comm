#!/usr/bin/env python3
"""Freeze the single CUDA-generator corrective readiness run."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CARD = ROOT / "configs/v3/gate3/data_cards/primitive_local_composition_slot_readiness_v1.json"
SOURCE_SPEC = ROOT / "configs/v3/gate3/primitive_local_composition_slot_readiness_v1.json"
SOURCE_RUN = ROOT / "results/gate3_semantics/gate3_20260903_primitive_local_composition_slot_readiness_v1_seed0"
CARD = ROOT / "configs/v3/gate3/data_cards/primitive_local_composition_slot_readiness_v1r.json"
SPEC = ROOT / "configs/v3/gate3/primitive_local_composition_slot_readiness_v1r.json"
RUN_ID = "gate3_20260903_primitive_local_composition_slot_readiness_v1r_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_LOCAL_COMPOSITION_SLOT_READINESS_V1R"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("local composition-slot V1R card/spec already exists")
    failed = _load(SOURCE_RUN / "RUN_STATE.json")
    expected_error = "RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'"
    if failed.get("state") != "FAILED" or failed.get("error") != expected_error:
        raise RuntimeError("local composition-slot V1R requires the exact sealed V1 system failure")
    source_card = _load(SOURCE_CARD); source_spec = _load(SOURCE_SPEC)
    card = copy.deepcopy(source_card)
    card["card_id"] = "primitive_local_composition_slot_readiness_v1r"
    card["purpose"] = source_card["purpose"] + " Correct V1's CPU-generator/CUDA-output bridge without changing any model, data, loss, batch, invariant or resource gate."
    card["failure_policy"] = source_card["failure_policy"] + " V1R additionally fails unless the exact V1 generator-device failure and seal are bound."
    card["retention"] = source_card["retention"] + " Preserve V1 as immutable system-failure evidence."
    card["sampling"]["rule"] = source_card["sampling"]["rule"] + " The only V1R code change is CPU fixed-seed randperm followed by an explicit CUDA copy."
    approval = {
        "approved_at": "2026-09-03T14:42:00+08:00",
        "approved_by": "user-standing-authorization",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "confirmation_reference": "Standing authorization plus the pre-registered fail-closed implementation-corrective policy.",
        "scope": "One immutable V1R readiness corrective; identical science and resources, only the fixed-seed permutation device bridge changes.",
        "status": "APPROVED",
    }
    card["approval"] = approval; card["status"] = CARD_STATUS; _write(CARD, card)

    spec = copy.deepcopy(source_spec)
    spec["slug"] = "primitive_local_composition_slot_readiness_v1r"
    spec["config_path"] = str(CARD.relative_to(ROOT)); spec["data_card"] = str(CARD.relative_to(ROOT))
    spec["question"] = source_spec["question"] + " Can the identical audit finish after correcting only the V1 CPU-generator/CUDA-output bridge?"
    spec["method"] = source_spec["method"] + " V1R generates the fixed-seed slot permutation on CPU and copies it to CUDA."
    spec["fallback"] = card["failure_policy"]; spec["user_authorization"] = approval
    spec["acceptance_criteria"] = list(card["metrics_and_pre_registered_gates"].values())
    spec["expected_evidence"] = list(source_spec["expected_evidence"]) + [
        "Exact V1 failure state, traceback and seal bound as corrective provenance."
    ]
    spec["command"] = [
        "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
        "--why=Primitive local composition slot readiness V1R", "--mode=block",
        "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "1800s",
        "/usr/bin/env", f"PYTHONPATH={ROOT / 'src'}:{ROOT / 'tools/v3'}:{ROOT}",
        "PYTHONHASHSEED=0", "CUBLAS_WORKSPACE_CONFIG=:4096:8", PYTHON,
        "tools/v3/run_primitive_local_composition_slot_readiness_v1r.py",
        "--spec", str(SPEC), "--run-dir", str(ROOT / "results/gate3_semantics" / RUN_ID),
    ]
    frozen_inputs = {
        relative: _sha(ROOT / relative)
        for relative in source_spec["frozen_inputs"]
        if relative != str(SOURCE_CARD.relative_to(ROOT))
    }
    frozen_inputs[str(CARD.relative_to(ROOT))] = _sha(CARD)
    for path in (
        SOURCE_CARD, SOURCE_SPEC, SOURCE_RUN / "RUN_STATE.json",
        SOURCE_RUN / "metrics/summary.json", SOURCE_RUN / "logs/failure_traceback.log",
        SOURCE_RUN / "artifacts/evidence_sha256.txt",
    ):
        frozen_inputs[str(path.relative_to(ROOT))] = _sha(path)
    spec["frozen_inputs"] = frozen_inputs
    tools = copy.deepcopy(source_spec["frozen_tools"])
    additions = {
        "corrective_runner": "tools/v3/run_primitive_local_composition_slot_readiness_v1r.py",
        "corrective_test": "tests/v3/unit/test_primitive_local_composition_slot_readiness_v1r.py",
        "corrective_freezer": "tools/v3/freeze_primitive_local_composition_slot_readiness_v1r_spec.py",
    }
    tools.update({
        name: {"path": path, "sha256": _sha(ROOT / path)} for name, path in additions.items()
    })
    spec["frozen_tools"] = tools
    _write(SPEC, spec); print(CARD); print(SPEC)


if __name__ == "__main__":
    main()
