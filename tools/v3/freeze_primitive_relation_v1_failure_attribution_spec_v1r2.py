#!/usr/bin/env python3
"""Freeze the one TF32-off C07 evaluation plus attribution corrective."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_v1_failure_attribution_v1r.json"
OLD_SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_v1_failure_attribution_v1r.json"
FAILED = PROJECT_ROOT / "results/gate3_semantics/gate3_20260831_primitive_relation_v1_failure_attribution_v1r_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_v1_failure_attribution_v1r2.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_v1_failure_attribution_v1r2.json"
RUN_ID = "gate3_20260831_primitive_relation_v1_failure_attribution_v1r2_seed0"


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
        raise RuntimeError("primitive relation attribution V1R2 card/spec already exists")
    failed_state = json.loads((FAILED / "RUN_STATE.json").read_text())
    failed_summary = json.loads((FAILED / "metrics/summary.json").read_text())
    if failed_state.get("state") != "FAILED" or "subprocess failed" not in str(failed_summary.get("error")):
        raise RuntimeError("V1R2 requires the sealed metric-reproduction system failure")
    old_card = json.loads(OLD_CARD.read_text()); old_spec = json.loads(OLD_SPEC.read_text())
    card = copy.deepcopy(old_card)
    card["card_id"] = "primitive_relation_v1_failure_attribution_v1r2"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_V1_FAILURE_ATTRIBUTION_V1R2"
    card["purpose"] = (
        "Correct the discovered formal-evaluator cuDNN TF32 contract drift by re-evaluating all three frozen checkpoints on C07 with matmul and cuDNN TF32 disabled, then attribute proposal versus relation failure under the same corrected setting."
    )
    card["approval"]["scope"] = (
        "One immutable C07-only TF32-off corrective evaluation plus failure attribution: 10 parents, 30 tasks, "
        "64644 sequences, three frozen checkpoints, 581796 total inference rows; zero optimizer, checkpoint change, "
        "C08/C09/C10, graph or M-TARE."
    )
    card["approval"]["confirmation_reference"] = (
        "User delegated autonomous best-evidence execution; V1R1 sealed evidence shows the original evaluator omitted the declared cuDNN TF32-off setting, so the strongest in-plan option is a C07-only numerical-contract corrective before attribution."
    )
    counts = card["sampling"]["structure_event_counts"]
    counts["corrective_evaluation_inference_rows"] = 387864
    counts["attribution_inference_rows"] = 193932
    counts["model_inference_rows"] = 581796
    card["leakage_audit"]["model_inference_count"] = 581796
    card["sampling"]["rule"] = (
        "First reproduce the complete C07 gate with both matmul and cuDNN TF32 disabled using two frozen passes per seed; then read every C07 row once more per seed for actual-versus-oracle attribution. No sampling, deletion, checkpoint/threshold reuse from C08 or raw pair persistence."
    )
    gates = card["metrics_and_pre_registered_gates"]
    gates["implementation"] = (
        "Eleven frozen CPU tests pass; corrective evaluator explicitly disables matmul and cuDNN TF32 and cannot construct a C08 loader. Attribution exactly reproduces the corrected C07 counts."
    )
    gates["population"] = (
        "Exactly 64644 rows and 30 tasks per seed; 387864 corrected-evaluation plus 193932 attribution inference rows; three selected epochs unchanged; C08 rows read zero."
    )
    gates["numerical_corrective"] = (
        "Record corrected-versus-old C07 results without treating old TF32-on counts as the reproduction target. All subsequent attribution uses only the corrected TF32-off metrics."
    )
    card["estimated_cost"]["wall_time_hours"] = 1.5
    card["retention"] = (
        "Retain corrected per-seed C07 metrics, per-seed/per-task attribution, actual/oracle pair statistics, PNG/PDF/SVG/source, old-versus-corrected provenance, environment, logs, RUN_STATE and seal."
    )
    write(CARD, card)

    spec = copy.deepcopy(old_spec)
    spec["slug"] = "primitive_relation_v1_failure_attribution_v1r2"
    spec["data_card"] = str(CARD.relative_to(PROJECT_ROOT)); spec["config_path"] = str(CARD.relative_to(PROJECT_ROOT))
    spec["question"] = (
        "Under the declared TF32-off evaluation contract, does frozen V1 still fail C07, and is any remaining attachment failure dominated by proposal over-activation or by relation logits themselves?"
    )
    spec["method"] = (
        "Run a C07-only two-pass-per-seed corrective evaluation with matmul/cuDNN TF32 disabled, then replay each same checkpoint once for actual-versus-Teacher-matched proposal-oracle attribution; no C08 or model changes."
    )
    spec["fallback"] = "Any corrected-metric, attribution, population or isolation failure stops; no return to TF32-on counts, C08, retraining or tolerance relaxation."
    spec["user_authorization"] = card["approval"]
    spec["acceptance_criteria"] = list(card["metrics_and_pre_registered_gates"].values())
    spec["expected_counts"]["unit_tests"] = 11
    spec["expected_counts"]["corrective_evaluation_inference_rows"] = 387864
    spec["expected_counts"]["attribution_inference_rows"] = 193932
    spec["expected_counts"]["model_inference_rows"] = 581796
    spec["estimated_cost"] = card["estimated_cost"]
    spec["corrected_c07_tf32_evaluation"] = True
    spec["command"][-5] = "tools/v3/run_primitive_relation_v1_failure_attribution_v1r2.py"
    spec["command"][-3] = str(SPEC)
    spec["command"][-1] = str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")

    frozen_inputs = {}
    for relative in old_spec["frozen_inputs"]:
        path = PROJECT_ROOT / relative
        if path == OLD_CARD:
            path = CARD
        frozen_inputs[str(path.relative_to(PROJECT_ROOT))] = sha(path)
    for path in (FAILED / "RUN_STATE.json", FAILED / "metrics/summary.json", FAILED / "artifacts/evidence_sha256.txt", FAILED / "logs/01_attribution.log"):
        frozen_inputs[str(path.relative_to(PROJECT_ROOT))] = sha(path)
    spec["frozen_inputs"] = frozen_inputs
    tools = {name: record["path"] for name, record in old_spec["frozen_tools"].items()}
    tools.update({
        "runner": "tools/v3/run_primitive_relation_v1_failure_attribution_v1r2.py",
        "base_runner": "tools/v3/run_primitive_relation_v1_failure_attribution.py",
        "freezer": "tools/v3/freeze_primitive_relation_v1_failure_attribution_spec_v1r2.py",
        "tf32_corrective_evaluator": "tools/v3/evaluate_primitive_relation_three_seed_c07_tf32_corrective_v1.py",
    })
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()
    }
    write(SPEC, spec); print(SPEC)


if __name__ == "__main__":
    main()
