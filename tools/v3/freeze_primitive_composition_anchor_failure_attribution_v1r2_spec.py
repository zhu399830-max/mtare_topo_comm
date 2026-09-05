#!/usr/bin/env python3
"""Freeze the numerical-contract corrective for C07 failure attribution."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CARD = ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_failure_attribution_v1r.json"
SOURCE_SPEC = ROOT / "configs/v3/gate3/primitive_composition_anchor_failure_attribution_v1r.json"
SOURCE_RUN = ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_failure_attribution_v1r_seed0"
CARD = ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_failure_attribution_v1r2.json"
SPEC = ROOT / "configs/v3/gate3/primitive_composition_anchor_failure_attribution_v1r2.json"
RUN_ID = "gate3_20260903_primitive_composition_anchor_failure_attribution_v1r2_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_FAILURE_ATTRIBUTION_V1R2"
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
        raise RuntimeError("composition-anchor attribution V1R2 card/spec already exists")
    state = _load(SOURCE_RUN / "RUN_STATE.json")
    log = (SOURCE_RUN / "logs/01_attribution.log").read_text(encoding="utf-8")
    if (
        state.get("state") != "FAILED"
        or "formal metric reproduction drift: true_positive" not in log
        or "--anchor-root" not in log
        or "artifacts/materialized/anchor_targets" not in log
    ):
        raise RuntimeError("V1R2 requires the exact post-seed0 numerical-contract mismatch")
    source_card = _load(SOURCE_CARD); source_spec = _load(SOURCE_SPEC)
    card = copy.deepcopy(source_card)
    card["card_id"] = "primitive_composition_anchor_failure_attribution_v1r2"
    card["purpose"] = (
        "Execute the unchanged C07 attribution under an explicit numerical state matching the "
        "actual fresh-process state of the sealed source evaluator."
    )
    card["corrective_scope"] = (
        "V1R fixed the input path and processed seed0, but its executor changed cuDNN TF32 from the "
        "source evaluator's actual true value to false and enabled deterministic algorithms. V1R2 "
        "explicitly freezes the source state: CUDA matmul TF32 false, cuDNN TF32 true, float32 precision "
        "highest and deterministic algorithms false. No data, model, threshold, metric or decision changes."
    )
    card["failure_policy"] = source_card["failure_policy"]
    card["metrics_and_pre_registered_gates"] = dict(source_card["metrics_and_pre_registered_gates"])
    card["metrics_and_pre_registered_gates"]["numerical_reproduction"] = (
        "The evaluator explicitly sets the source evaluator's actual fresh-process numerical state and "
        "must exactly reproduce TP/FP/FN/selected-pair counts and F1 for all three seeds."
    )
    approval = copy.deepcopy(source_card["approval"])
    approval["scope"] = "One immutable numerical-contract V1R2 C07 attribution corrective; no training, C08+, graph or M-TARE."
    card["approval"] = approval; card["status"] = CARD_STATUS
    _write(CARD, card)
    spec = copy.deepcopy(source_spec)
    spec.update({
        "slug": "primitive_composition_anchor_failure_attribution_v1r2",
        "config_path": str(CARD.relative_to(ROOT)), "data_card": str(CARD.relative_to(ROOT)),
        "method": source_spec["method"] + " V1R2 explicitly reproduces the source evaluator numerical state.",
        "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=Composition anchor C07 attribution V1R2",
            "--mode=block", "/usr/bin/timeout", "--signal=INT", "--kill-after=120s", "14400s",
            "/usr/bin/env", f"PYTHONPATH={ROOT / 'src'}:{ROOT / 'tools/v3'}:{ROOT}",
            "PYTHONHASHSEED=0", "CUBLAS_WORKSPACE_CONFIG=:4096:8", PYTHON,
            "tools/v3/run_primitive_composition_anchor_failure_attribution_v1r2.py",
            "--spec", str(SPEC), "--run-dir", str(ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    })
    frozen_inputs = dict(source_spec["frozen_inputs"])
    extra = [
        CARD, SOURCE_CARD, SOURCE_SPEC, SOURCE_RUN / "RUN_STATE.json",
        SOURCE_RUN / "metrics/summary.json", SOURCE_RUN / "logs/01_attribution.log",
        SOURCE_RUN / "artifacts/evidence_sha256.txt",
    ]
    frozen_inputs.update({str(path.relative_to(ROOT)): _sha(path) for path in extra})
    spec["frozen_inputs"] = frozen_inputs
    tools = dict(source_spec["frozen_tools"])
    tools["v1r_runner"] = tools.pop("runner")
    tools["runner"] = {
        "path": "tools/v3/run_primitive_composition_anchor_failure_attribution_v1r2.py",
        "sha256": _sha(ROOT / "tools/v3/run_primitive_composition_anchor_failure_attribution_v1r2.py"),
    }
    tools["executor_v1r2"] = {
        "path": "tools/v3/execute_primitive_composition_anchor_failure_attribution_v1r2.py",
        "sha256": _sha(ROOT / "tools/v3/execute_primitive_composition_anchor_failure_attribution_v1r2.py"),
    }
    tools["numerical_corrective_tests"] = {
        "path": "tests/v3/unit/test_primitive_composition_anchor_failure_attribution_v1r2.py",
        "sha256": _sha(ROOT / "tests/v3/unit/test_primitive_composition_anchor_failure_attribution_v1r2.py"),
    }
    tools["freezer"] = {
        "path": "tools/v3/freeze_primitive_composition_anchor_failure_attribution_v1r2_spec.py",
        "sha256": _sha(ROOT / "tools/v3/freeze_primitive_composition_anchor_failure_attribution_v1r2_spec.py"),
    }
    spec["frozen_tools"] = tools
    _write(SPEC, spec)
    print(CARD); print(SPEC)


if __name__ == "__main__":
    main()
