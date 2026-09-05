#!/usr/bin/env python3
"""Freeze the path-only corrective for C07 composition-anchor attribution."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CARD = ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_failure_attribution_v1.json"
SOURCE_SPEC = ROOT / "configs/v3/gate3/primitive_composition_anchor_failure_attribution_v1.json"
SOURCE_RUN = ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_failure_attribution_v1_seed0"
ANCHOR_RUN = ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_target_sidecar_v1r_seed0"
CARD = ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_failure_attribution_v1r.json"
SPEC = ROOT / "configs/v3/gate3/primitive_composition_anchor_failure_attribution_v1r.json"
RUN_ID = "gate3_20260903_primitive_composition_anchor_failure_attribution_v1r_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_FAILURE_ATTRIBUTION_V1R"
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
        raise RuntimeError("composition-anchor attribution V1R card/spec already exists")
    state = _load(SOURCE_RUN / "RUN_STATE.json")
    log = (SOURCE_RUN / "logs/01_attribution.log").read_text(encoding="utf-8")
    if (
        state.get("state") != "FAILED"
        or state.get("duration_seconds", 99) > 10
        or "composition-anchor sidecars are empty or task-misaligned" not in log
        or "artifacts/anchors" not in log
    ):
        raise RuntimeError("V1R requires the exact zero-inference path-binding failure")
    source_card = _load(SOURCE_CARD); source_spec = _load(SOURCE_SPEC)
    card = copy.deepcopy(source_card)
    card["card_id"] = "primitive_composition_anchor_failure_attribution_v1r"
    card["purpose"] = (
        "Execute the unchanged C07 attribution after binding --anchor-root to the sealed "
        "artifacts/materialized/anchor_targets directory instead of the nonexistent artifacts/anchors directory."
    )
    card["corrective_scope"] = (
        "V1 stopped before loading a single row because its runner used the wrong sidecar directory name. "
        "V1R changes only that path binding and immutable run/card identity; model, data, metrics, thresholds, "
        "counterfactuals and resources remain unchanged."
    )
    card["failure_policy"] = source_card["failure_policy"]
    approval = copy.deepcopy(source_card["approval"])
    approval["scope"] = "One immutable path-only V1R C07 attribution corrective; no C08+, graph, training or M-TARE."
    card["approval"] = approval; card["status"] = CARD_STATUS
    _write(CARD, card)
    spec = copy.deepcopy(source_spec)
    spec.update({
        "slug": "primitive_composition_anchor_failure_attribution_v1r",
        "config_path": str(CARD.relative_to(ROOT)), "data_card": str(CARD.relative_to(ROOT)),
        "question": source_spec["question"],
        "method": source_spec["method"] + " V1R changes only the anchor sidecar root binding.",
        "user_authorization": approval,
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=Composition anchor C07 attribution V1R",
            "--mode=block", "/usr/bin/timeout", "--signal=INT", "--kill-after=120s", "14400s",
            "/usr/bin/env", f"PYTHONPATH={ROOT / 'src'}:{ROOT / 'tools/v3'}:{ROOT}",
            "PYTHONHASHSEED=0", "CUBLAS_WORKSPACE_CONFIG=:4096:8", PYTHON,
            "tools/v3/run_primitive_composition_anchor_failure_attribution_v1r.py",
            "--spec", str(SPEC), "--run-dir", str(ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    })
    frozen_inputs = dict(source_spec["frozen_inputs"])
    extra = [
        CARD, SOURCE_CARD, SOURCE_SPEC, SOURCE_RUN / "RUN_STATE.json",
        SOURCE_RUN / "metrics/summary.json", SOURCE_RUN / "logs/01_attribution.log",
        SOURCE_RUN / "artifacts/evidence_sha256.txt",
        ANCHOR_RUN / "artifacts/materialized/task_manifest.json",
        ANCHOR_RUN / "artifacts/evidence_sha256.txt",
    ]
    frozen_inputs.update({str(path.relative_to(ROOT)): _sha(path) for path in extra})
    spec["frozen_inputs"] = frozen_inputs
    tools = dict(source_spec["frozen_tools"])
    tools["source_runner"] = tools.pop("runner")
    tools["runner"] = {
        "path": "tools/v3/run_primitive_composition_anchor_failure_attribution_v1r.py",
        "sha256": _sha(ROOT / "tools/v3/run_primitive_composition_anchor_failure_attribution_v1r.py"),
    }
    tools["corrective_tests"] = {
        "path": "tests/v3/unit/test_primitive_composition_anchor_failure_attribution_v1r.py",
        "sha256": _sha(ROOT / "tests/v3/unit/test_primitive_composition_anchor_failure_attribution_v1r.py"),
    }
    tools["freezer"] = {
        "path": "tools/v3/freeze_primitive_composition_anchor_failure_attribution_v1r_spec.py",
        "sha256": _sha(ROOT / "tools/v3/freeze_primitive_composition_anchor_failure_attribution_v1r_spec.py"),
    }
    spec["frozen_tools"] = tools
    _write(SPEC, spec)
    print(CARD); print(SPEC)


if __name__ == "__main__":
    main()
