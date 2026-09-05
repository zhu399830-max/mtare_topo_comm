#!/usr/bin/env python3
"""Freeze the batch-rank-only corrective attribution V1R."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_sparse_port_failure_attribution_v1.json"
OLD_SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_sparse_port_failure_attribution_v1.json"
FAILED = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_sparse_port_failure_attribution_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_sparse_port_failure_attribution_v1r.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_sparse_port_failure_attribution_v1r.json"
RUN_ID = "gate3_20260902_primitive_relation_sparse_port_failure_attribution_v1r_seed0"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("sparse-port attribution V1R card/spec already exists")
    state = json.loads((FAILED / "RUN_STATE.json").read_text())
    summary = json.loads((FAILED / "metrics/summary.json").read_text())
    if state.get("state") != "FAILED" or summary.get("model_inference_rows") != 0:
        raise RuntimeError("V1R requires the sealed zero-population V1 system failure")
    if "sparse-port attribution subprocess failed" not in str(summary.get("error")):
        raise RuntimeError("V1 failure class drift")
    log = (FAILED / "logs/01_attribution.log").read_text()
    if "src [1, 32]" not in log or "index [128, 32]" not in log:
        raise RuntimeError("V1R batch-rank failure evidence drift")

    card = copy.deepcopy(json.loads(OLD_CARD.read_text()))
    card["card_id"] = "primitive_relation_sparse_port_failure_attribution_v1r"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_SPARSE_PORT_FAILURE_ATTRIBUTION_V1R"
    card["purpose"] = (
        "Repeat the identical C07-only sparse-port failure attribution after the sole implementation correction expands the deterministic rank source from [1,32] to the formal batch shape [B,32]."
    )
    card["approval"]["scope"] = (
        "One immutable batch-rank-only corrective of the same zero-training C07 audit: 10 parents, 30 tasks, 64644 rows per frozen seed, 193932 inference rows; no data, Teacher, model, score, decoder, metric, threshold, C08/C09/C10, graph or M-TARE change."
    )
    card["approval"]["confirmation_reference"] = (
        "The sealed V1 stopped on its first formal batch because torch.scatter_ did not broadcast [1,32] rank values to [128,32]. The user's standing authorization selects the exact batch expansion plus a batch-128 regression test as the minimum in-plan corrective."
    )
    card["metrics_and_pre_registered_gates"]["implementation"] = (
        "Nineteen frozen unit tests pass, including a 128-row Teacher-cardinality rank regression; formal V2 attachment counts must still reproduce exactly before diagnostics."
    )
    card["failure_policy"] = (
        "Any failure after the single rank-source batch expansion stops. No additional implementation repair, data/model/Teacher/score/decoder/metric change, C08, graph or planner compensation is allowed inside V1R."
    )
    card["retention"] += " Bind and retain the sealed V1 zero-inference failure and batch-shape traceback."
    write(CARD, card)

    spec = copy.deepcopy(json.loads(OLD_SPEC.read_text()))
    spec["slug"] = "primitive_relation_sparse_port_failure_attribution_v1r"
    spec["data_card"] = str(CARD.relative_to(PROJECT_ROOT))
    spec["config_path"] = str(CARD.relative_to(PROJECT_ROOT))
    spec["question"] = (
        "After the exact batch-rank expansion, what mechanism explains the frozen sparse-port model's 0/3 C07 relation failure?"
    )
    spec["method"] = (
        "Identical V1 C07-only attribution with one implementation correction: expand deterministic slot ranks to [B,32] before scatter; all model, data, masks, decoders, scores and decision order remain frozen."
    )
    spec["fallback"] = card["failure_policy"]
    spec["user_authorization"] = card["approval"]
    spec["acceptance_criteria"] = list(card["metrics_and_pre_registered_gates"].values())
    spec["expected_counts"]["unit_tests"] = 19
    spec["estimated_cost"] = card["estimated_cost"]
    for index, value in enumerate(spec["command"]):
        if value == "tools/v3/run_primitive_relation_sparse_port_failure_attribution_v1.py":
            spec["command"][index] = "tools/v3/run_primitive_relation_sparse_port_failure_attribution_v1r.py"
        elif value == str(OLD_SPEC):
            spec["command"][index] = str(SPEC)
        elif value.endswith("gate3_20260902_primitive_relation_sparse_port_failure_attribution_v1_seed0"):
            spec["command"][index] = str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")

    frozen_inputs = {}
    old_card_relative = str(OLD_CARD.relative_to(PROJECT_ROOT))
    for relative in spec["frozen_inputs"]:
        path = CARD if relative == old_card_relative else PROJECT_ROOT / relative
        frozen_inputs[str(path.relative_to(PROJECT_ROOT))] = sha(path)
    for path in (
        FAILED / "RUN_STATE.json", FAILED / "metrics/summary.json",
        FAILED / "artifacts/evidence_sha256.txt", FAILED / "logs/01_attribution.log",
    ):
        frozen_inputs[str(path.relative_to(PROJECT_ROOT))] = sha(path)
    spec["frozen_inputs"] = frozen_inputs

    tools = {name: record["path"] for name, record in spec["frozen_tools"].items()}
    tools.update({
        "attribution_module": "src/mtare_topo/evaluation/primitive_relation_sparse_port_failure_attribution.py",
        "attribution_tests": "tests/v3/unit/test_primitive_relation_sparse_port_failure_attribution.py",
        "runner": "tools/v3/run_primitive_relation_sparse_port_failure_attribution_v1r.py",
        "base_runner": "tools/v3/run_primitive_relation_sparse_port_failure_attribution_v1.py",
        "freezer": "tools/v3/freeze_primitive_relation_sparse_port_failure_attribution_v1r_spec.py",
    })
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
