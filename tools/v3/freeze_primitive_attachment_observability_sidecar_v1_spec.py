#!/usr/bin/env python3
"""Freeze one compact C01--C07 endpoint-observability sidecar export."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_attachment_teacher_observability_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_attachment_observability_sidecar_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_attachment_observability_sidecar_v1.json"
RUN_ID = "gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
AUDIT = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_teacher_observability_v1_seed0"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CARD.exists() or SPEC.exists(): raise RuntimeError("attachment sidecar card/spec already exists")
    summary = json.loads((AUDIT / "metrics/summary.json").read_text())
    state = json.loads((AUDIT / "RUN_STATE.json").read_text())
    if state.get("state") != "COMPLETED" or summary.get("diagnosis") != "WINDOW_LEVEL_HIDDEN_ENDPOINT_SUPERVISION_CORRECTABLE_BY_OBSERVABILITY_MASK":
        raise RuntimeError("source observability decision drift")
    fit = summary["split"]["fit"]; c07 = summary["split"]["c07"]
    fit_supported = int(fit["support_bands"]["0.25m"]["both_endpoints"])
    c07_supported = int(c07["support_bands"]["0.25m"]["both_endpoints"])
    if (fit_supported, c07_supported) != (2_782_487, 442_936): raise RuntimeError("supported-positive counts drift")

    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card["card_id"] = "primitive_attachment_observability_sidecar_v1"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_ATTACHMENT_OBSERVABILITY_SIDECAR_V1"
    card["purpose"] = "Materialize the smallest bit-exact endpoint-support sidecar needed to convert hidden P1b attachment pairs from false negatives/forced positives into unknown relation targets."
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-02T14:30:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["teacher_generation"],
        "scope": "One immutable C01--C07-only 0.25 m endpoint-observability sidecar for 210 geometry tasks and 491196 sequences; no duplicated LiDAR, no C08/C09/C10, model inference, graph or M-TARE.",
        "confirmation_reference": "The user authorized uninterrupted best-in-plan execution. The immediately preceding sealed audit pre-registered this sidecar when identity coverage exceeded 0.95 in fit and C07.",
    }
    card["sampling"]["rule"] = "For every existing P1b sequence, store exactly 64 endpoint-support bits using the frozen 0.25 m band and the existing 0.025 m sampled swept axes. Preserve source sequence identity; do not copy LiDAR, geometry targets or relation labels."
    card["sampling"]["structure_event_counts"].update({
        "fit_supported_positive_pairs": fit_supported,
        "c07_supported_positive_pairs": c07_supported,
        "fit_hidden_positive_pairs": 3_301_484 - fit_supported,
        "c07_hidden_positive_pairs": 525_077 - c07_supported,
        "sidecar_tasks": 210,
    })
    card["teacher"] = {
        "source": "Sealed P1a realized endpoints plus P1b five-frame cropped axes; 0.25 m support band inherited unchanged from the approved observability audit.",
        "valid_mask": "An ordered attachment candidate is valid only when both candidate endpoint bits are one and the endpoints belong to different active primitives. Any pair with a zero endpoint bit is unknown and must never enter positive or negative attachment loss.",
        "planner_consistency_plan": "The sidecar authorizes only masked relation-model readiness. Deployed relation confidence must later include learned endpoint-evidence confidence, and graph edges remain traversal-only.",
        "student_forbidden_inputs": "The bitmask is Teacher-only supervision/loss validity. Construction identity, true endpoint coordinates, P1a pose, TNG and future frames remain forbidden model inputs.",
    }
    card["metrics_and_pre_registered_gates"] = {
        "population": "Exactly 210 C01--C07 geometry tasks and 491196 sequences; fit/C07 426552/64644; C08+ zero.",
        "contract": "Every sidecar contains only source sequence indices and bit-packed [32,2] endpoint support; pack/unpack is bit exact and observed endpoints are a subset of active primitives.",
        "positive_counts": "Supported fit/C07 positive pairs exactly reproduce 2782487/442936; hidden positives 518997/82141 remain unknown, never negative.",
        "negative_counts": "Both splits contain a nonempty observable negative endpoint-pair population for relation discrimination.",
        "integrity": "Every sidecar binds the sealed source P1b shard hash and receives its own tree hash; six unit tests pass.",
        "resources": "CPU wall time <=2 h, host RSS <=16 GiB, output <=0.2 GiB, zero GPU/model/optimizer/checkpoint.",
        "decision": "PASS allows only masked relation-model readiness; it does not revive the failed V2 relation checkpoint or unlock C08/graph.",
    }
    card["estimated_cost"] = {"compute": "Serial CPU/Zarr compact Teacher-sidecar export; no GPU.", "gpu": 0, "host_ram_gb": 16, "wall_time_hours": 1.0, "disk_gb": .2}
    card["failure_policy"] = "Any source/audit hash, count, bit round-trip, active-mask, sequence identity, resource or isolation drift fails closed; do not change the 0.25 m band inside the run."
    card["retention"] = "Retain compact per-task sidecars, task manifest, counts, source audit figure, environment, logs, RUN_STATE and SHA-256 seal."
    write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    expected = {
        "geometry_tasks": 210, "fit_sequences": 426_552, "c07_sequences": 64_644,
        "fit_supported_positive_pairs": fit_supported, "c07_supported_positive_pairs": c07_supported,
        "unit_tests": 6, "optimizer_steps": 0, "model_inference_rows": 0,
        "c08_rows_read": 0, "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "slug": "primitive_attachment_observability_sidecar_v1",
        "date": "20260902", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "teacher_generation", "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)), "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": "Can the audited endpoint support be materialized losslessly as a compact Teacher-only validity mask for the next learned relation model?",
        "method": "Compute endpoint-to-observed-crop gap exactly as in the sealed audit, threshold once at 0.25 m, bit-pack 64 endpoint flags and bind every sidecar to its source P1b shard and sequence indices.",
        "baseline": "Unmasked P1b attachment supervision, where 15.7% of window positives lack dual endpoint evidence and all active endpoint pairs are treated as known negatives otherwise.",
        "fallback": "Any failure stops before model readiness. No direct modification of immutable P1b, duplicated sensor export, C08, graph or planner fallback is allowed.",
        "user_authorization": card["approval"], "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": expected, "estimated_cost": card["estimated_cost"],
        "expected_evidence": ["210 compact hashed sidecars with bit-exact source sequence binding.", "Exact supported/hidden positive and observable negative counts per split.", "Six tests, task manifest, source figure, environment, logs, RUN_STATE and SHA-256 seal."],
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=Primitive attachment observability sidecar", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "7200s", "/usr/bin/env",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}", "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python",
            "tools/v3/run_primitive_attachment_observability_sidecar_v1.py", "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [CARD, *[run / name for run in (P1A, P1B, AUDIT) for name in ("RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt")], P1A / "artifacts/task_manifest.json", P1B / "artifacts/task_manifest.json", AUDIT / "metrics/attachment_observability.json"]
    spec["frozen_inputs"] = {str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs}
    tools = {
        "sidecar_module": "src/mtare_topo/data/primitive_attachment_observability_sidecar.py",
        "sidecar_tests": "tests/v3/unit/test_primitive_attachment_observability_sidecar.py",
        "audit_module": "src/mtare_topo/evaluation/primitive_attachment_observability.py",
        "audit_tests": "tests/v3/unit/test_primitive_attachment_observability.py",
        "materialization": "src/mtare_topo/data/primitive_relation_materialization.py",
        "runner": "tools/v3/run_primitive_attachment_observability_sidecar_v1.py",
        "freezer": "tools/v3/freeze_primitive_attachment_observability_sidecar_v1_spec.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()}
    write(SPEC, spec); print(SPEC)


if __name__ == "__main__": main()
