#!/usr/bin/env python3
"""Freeze endpoint relation metric real-batch readiness."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CARD = ROOT / "configs/v3/gate3/data_cards/primitive_local_composition_slot_readiness_v1.json"
CARD = ROOT / "configs/v3/gate3/data_cards/primitive_endpoint_relation_metric_readiness_v1.json"
SPEC = ROOT / "configs/v3/gate3/primitive_endpoint_relation_metric_readiness_v1.json"
RUN_ID = "gate3_20260904_primitive_endpoint_relation_metric_readiness_v1_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_ENDPOINT_RELATION_METRIC_READINESS_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBS = ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
SOURCE_MODELS = ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
ATTRIBUTION = ROOT / "results/gate3_semantics/gate3_20260904_primitive_local_composition_slot_failure_attribution_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _load(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CARD.exists() or SPEC.exists(): raise RuntimeError("endpoint metric readiness card/spec already exists")
    attribution = _load(ATTRIBUTION / "metrics/summary.json")
    if attribution.get("scientific_pass") is not True or attribution.get("decision") != "SLOT_IDENTITY_AND_CONFIDENCE_NOT_CROSS_SEED_SAFE":
        raise RuntimeError("endpoint metric readiness requires exact local-slot attribution")
    card = copy.deepcopy(_load(SOURCE_CARD))
    card.update({
        "card_id": "primitive_endpoint_relation_metric_readiness_v1", "status": CARD_STATUS,
        "purpose": "Zero-training real-batch proof that a no-arbitrary-slot endpoint embedding can receive same-composition/different/overlap gradients and decode one-threshold complete-link clusters without Teacher data in forward.",
        "teacher_source": "One fixed 128-row C01 fit batch from sealed P1a/P1b/observability. Construction clique labels and overlap negatives are used only after forward to evaluate losses.",
        "estimated_cost": {"compute": "One deterministic CUDA forward/backward, permutation/yaw/repeat probes and complete-link decode.", "disk_gb": .1, "gpu": 1, "gpu_memory_gb": 16, "host_ram_gb": 4, "wall_time_hours": .25},
        "failure_policy": "Any gradient, permutation, yaw, symmetry, deterministic clustering, source hash, environment, resource or leakage failure stops. Readiness makes no accuracy claim and cannot open C08 or graph.",
        "retention": "Keep test log, real-batch losses/gradients/contracts, implementation figure PNG/PDF/SVG, environment, RUN_STATE and SHA-256 seal.",
    })
    card["sampling"] = {
        "raw_frame_count": 640, "effective_sample_count": 128,
        "effective_structure_event_count": 850,
        "independent_sampling_units": "One fixed C01 topology/geometry task and 128 five-frame sequences; implementation probe only, not an accuracy estimate.",
        "rule": "Read rows 0:128 from the sealed S01_flat_tree_small_C01__c1_mixed fit task exactly once; inspect only C07 manifest counts and perform zero C07 forward.",
        "spatial_interval_m": 1.0, "temporal_window_frames": 5,
        "structure_event_counts": {"batch_rows": 128, "clusters": 850, "clustered_endpoints": 1770, "dustbin_endpoints": 99, "overlap_upper_pairs": 2520, "optimizer_steps": 0, "c07_forward_rows": 0, "c08_rows": 0},
    }
    card["split"] = {
        "fit": "Only the fixed C01 batch may be loaded for forward/backward implementation checks.",
        "selection": "C07 arrays are not loaded; only sealed manifest counts are checked.",
        "development_transfer": "C08 remains unopened.", "strict_test": "C08/C09/C10 and M-TARE are forbidden.",
        "world_disjoint": True, "trajectory_disjoint": True,
        "historical_pollution_audit": "The architecture and all contracts were written after formal slot attribution but before any endpoint-metric training or C07 inference.",
    }
    card["teacher"] = {
        "source": "Sealed construction composition cliques and endpoint observability for one C01 batch.",
        "labels": "Same-clique pair positives; different-clique, singleton dustbin and disconnected-overlap negatives.",
        "valid_mask": "Observed cross-primitive upper-triangle pairs only.",
        "student_forbidden_inputs": "Teacher labels, world/node/primitive identity, absolute pose, future frames and C07+ never enter forward.",
        "planner_consistency_plan": "A future accepted cluster proposes a local relation only; graph edges still require physical traversal.",
    }
    card["leakage_audit"] = {"model_inference_count": 128, "optimizer_step_count": 0, "future_sensor_frames_excluded": True, "absolute_pose_not_retained_in_student_representation": True, "test_excluded_from_supervised_training": True, "test_excluded_from_ssl": True, "test_excluded_from_normalization": True, "test_excluded_from_teacher_calibration": True, "test_excluded_from_threshold_calibration": True, "test_excluded_from_checkpoint_selection": True, "test_excluded_from_augmentation_tuning": True, "mtare_benchmark_excluded": True}
    card["metrics_and_pre_registered_gates"] = {
        "model": "426818 trainable head parameters/34 tensors and 2635631 frozen backbone parameters; O(E) endpoint embeddings and symmetric O(E^2) metric only at scoring.",
        "gradient": "All four losses finite; every head tensor has finite nonzero gradient; all backbone gradients absent and state unchanged.",
        "invariance": "Repeat bit-exact; primitive permutation and sensor-yaw errors <=3e-5; pair score exact symmetric.",
        "clustering": "One-threshold complete-link decode is deterministic and unit tests refuse chain merges.",
        "population": "Exact fit/C07 manifests 426552/64644; fixed batch128 contains nonempty clique, dustbin and overlap supervision; zero C07/C08 forward.",
        "resources": "CUDA allocated/reserved/process <=16 GiB and host RSS <=4 GiB.",
    }
    approval = {"approved_at": "2026-09-04T13:00:00+08:00", "approved_by": "user-standing-authorization", "authorized_gates": [3], "authorized_operations": ["audit"], "confirmation_reference": "The user instructed uninterrupted best-plan execution and the formal slot attribution pre-registered this minimal next readiness.", "scope": "One immutable C01 real-batch endpoint relation metric readiness; no training, C07 model forward, C08 or graph.", "status": "APPROVED"}
    card["approval"] = approval; _write(CARD, card)
    run_dir = ROOT / "results/gate3_semantics" / RUN_ID
    spec = {
        "schema_version": "v3_run_spec_v1", "slug": "primitive_endpoint_relation_metric_readiness_v1", "date": "20260904", "seed": 0, "gate": 3, "execution_phase": 3, "operation": "audit", "working_directory": str(ROOT), "config_path": str(CARD.relative_to(ROOT)), "data_card": str(CARD.relative_to(ROOT)),
        "question": "Can a no-slot endpoint relation metric satisfy real-batch learning, invariance, conservative clustering and resource contracts before training?",
        "method": "Freeze the observable primitive backbone; map each endpoint to a normalized 64-D relation embedding; supervise balanced pair logistic, same-clique compactness, different-clique separation and overlap rejection; decode with one-threshold deterministic complete link.",
        "baseline": "The formally stopped arbitrary 32-slot/dustbin head, whose hard identity precision was 17.9-21.9 percent on C07.",
        "fallback": card["failure_policy"], "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {"fit_rows": 426552, "c07_rows": 64644, "real_batch_rows": 128, "head_parameters": 426818, "head_tensors": 34, "frozen_parameters": 2635631, "tests": 17, "optimizer_steps": 0, "c07_forward_rows": 0, "c08_rows": 0},
        "selected_real_data_hashes": {"observability": "2796eaf975cec424fa3af358205a7eba29469060093f4a0915251670b89e7c65", "sensor": "729b3bd4cad8611fefacdfaebaa84b48d48edd5d979c1efbd4b96eb7b929bbbb", "teacher": "3ec3dff382e34349b0e655a88970bbc9ead861cbeefdb9e85e410eae141d4070"},
        "estimated_cost": card["estimated_cost"], "expected_evidence": ["17 unit/contract tests and fixed real-batch four-loss gradients.", "Permutation/yaw/repeat/symmetry/complete-link contracts and resource measurements.", "Implementation PNG/PDF/SVG, source hashes, environment, logs, RUN_STATE and seal."],
        "command": ["/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=Endpoint relation metric readiness", "--mode=block", "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "900s", "/usr/bin/env", f"PYTHONPATH={ROOT / 'src'}:{ROOT / 'tools/v3'}:{ROOT}", "PYTHONHASHSEED=0", "CUBLAS_WORKSPACE_CONFIG=:4096:8", PYTHON, "tools/v3/run_primitive_endpoint_relation_metric_readiness_v1.py", "--spec", str(SPEC), "--run-dir", str(run_dir)],
    }
    inputs = [CARD, SOURCE_CARD, ATTRIBUTION / "RUN_STATE.json", ATTRIBUTION / "metrics/summary.json", ATTRIBUTION / "metrics/attribution/summary.json", ATTRIBUTION / "artifacts/evidence_sha256.txt", P1A / "RUN_STATE.json", P1A / "metrics/summary.json", P1A / "artifacts/task_manifest.json", P1A / "artifacts/evidence_sha256.txt", P1B / "RUN_STATE.json", P1B / "metrics/summary.json", P1B / "artifacts/task_manifest.json", P1B / "artifacts/evidence_sha256.txt", OBS / "RUN_STATE.json", OBS / "metrics/summary.json", OBS / "artifacts/task_manifest.json", OBS / "artifacts/evidence_sha256.txt", SOURCE_MODELS / "RUN_STATE.json", SOURCE_MODELS / "metrics/summary.json", SOURCE_MODELS / "artifacts/models/seed0/selected.pt", SOURCE_MODELS / "artifacts/evidence_sha256.txt"]
    spec["frozen_inputs"] = {str(path.relative_to(ROOT)): _sha(path) for path in inputs}
    tools = {"model": "src/mtare_topo/representation/primitive_endpoint_relation_metric_model.py", "loss": "src/mtare_topo/representation/primitive_endpoint_relation_metric_training.py", "decoder": "src/mtare_topo/representation/primitive_endpoint_relation_metric_decoding.py", "runner": "tools/v3/run_primitive_endpoint_relation_metric_readiness_v1.py", "freezer": "tools/v3/freeze_primitive_endpoint_relation_metric_readiness_v1_spec.py", "tests": "tests/v3/unit/test_primitive_endpoint_relation_metric.py", "target_tests": "tests/v3/unit/test_primitive_local_composition_slot_training.py", "backbone_tests": "tests/v3/unit/test_primitive_relation_observable_model.py", "readiness_helpers": "tools/v3/run_primitive_local_composition_slot_readiness_v1.py", "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py"}
    spec["frozen_tools"] = {name: {"path": path, "sha256": _sha(ROOT / path)} for name, path in tools.items()}; _write(SPEC, spec); print(CARD); print(SPEC)


if __name__ == "__main__": main()
