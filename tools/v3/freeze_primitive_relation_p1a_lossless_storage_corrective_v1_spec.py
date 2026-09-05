#!/usr/bin/env python3
"""Freeze the P1a storage-only corrective after the immutable source seals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_sensor_provenance_export_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_p1a_lossless_storage_corrective_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_p1a_lossless_storage_corrective_v1.json"
RUN_ID = "gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"


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
    summary = json.loads((SOURCE / "metrics/summary.json").read_text())
    state = json.loads((SOURCE / "RUN_STATE.json").read_text())
    checks = dict(summary.get("checks", {}))
    false_checks = sorted(name for name, passed in checks.items() if not passed)
    if state.get("state") != "COMPLETED" or state.get("error") is not None:
        raise RuntimeError("source P1a must be cleanly completed")
    if summary.get("overall_status") != "FAIL_PRIMITIVE_RELATION_P1A_SENSOR_PROVENANCE_EXPORT_V1" or false_checks != ["result_bytes_le_20gib"]:
        raise RuntimeError(f"storage corrective requires a resource-only source FAIL, got {false_checks}")
    source_card = json.loads((SOURCE / "config/data_card.json").read_text())
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-30T00:00:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["data_export"],
        "confirmation_reference": "User instructed continuous autonomous execution and evidence-optimal in-plan choices; this corrective only changes lossless physical storage after the frozen resource-only FAIL.",
        "scope": "One immutable C01-C08 lossless storage corrective over exactly the 240 sealed P1a shards; no rerendering, relabeling, frame deletion, C09/C10, model, graph, planner or M-TARE.",
    }
    card = {
        "schema_version": "v3_data_card_v1", "card_id": "primitive_relation_p1a_lossless_storage_corrective_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_LOSSLESS_STORAGE_CORRECTIVE", "approval": approval,
        "purpose": "Replace only the physical Zarr compressor so the complete P1a corpus satisfies its frozen 20 GiB storage resource contract.",
        "worlds": source_card["worlds"], "source": source_card["source"], "trajectories": source_card["trajectories"],
        "sampling": {
            "independent_sampling_units": "The exact same 80 topology parents and 240 paired-realization shards as sealed P1a.",
            "raw_frame_count": 757290, "effective_sample_count": 80, "effective_structure_event_count": 24117,
            "ray_count": 8723980800, "spatial_interval_m": 1.0, "temporal_window_frames": 5,
            "structure_event_counts": {"directed_traversals": 16078, "planned_five_frame_sequences": 564378, "source_primitives": 8039, "variant_primitives": 24117},
            "rule": "Read every existing logical array chunk and write it with Blosc zstd level 9 byte-shuffle; reread and compare raw dtype bytes chunk-by-chunk. No samples or labels may change.",
        },
        "split": source_card["split"],
        "teacher": {
            "source": "Unchanged sealed P1a analytic CSG provenance.",
            "labels": "Unchanged range, validity, full primitive-source membership and frame metadata arrays.",
            "student_forbidden_inputs": "Unchanged from sealed P1a; absolute pose metadata is retained only for Teacher/evaluation and remains forbidden as model input.",
            "valid_mask": "Unchanged sealed P1a definition: valid only within the frozen near/50m range after CSG path/surface qualification; code0 iff invalid.",
            "planner_consistency_plan": "No graph or planner executes in this storage-only corrective.",
        },
        "leakage_audit": {
            "optimizer_step_count": 0, "model_inference_count": 0, "new_render_count": 0,
            "test_excluded_from_supervised_training": True, "test_excluded_from_ssl": True,
            "test_excluded_from_normalization": True, "test_excluded_from_augmentation_tuning": True,
            "test_excluded_from_teacher_calibration": True, "test_excluded_from_threshold_calibration": True,
            "test_excluded_from_checkpoint_selection": True, "mtare_benchmark_excluded": True,
        },
        "metrics_and_pre_registered_gates": {
            "source": "The source run must have exactly one failed check: result_bytes_le_20gib; all scientific/data checks must pass.",
            "population": "Exactly 240 shards, 80 parents, 757,290 frames, 8,723,980,800 rays, 24,117 primitives and 402 corrections.",
            "bit_exact": "All 11 arrays in every shard must match source dtype bytes chunk-by-chunk after write/reopen.",
            "metadata": "Group/array metadata, codebooks and construction documents remain exact; source and corrected shard trees are recorded.",
            "resources": "Corrected compressed shard bytes <=20 GiB and lower than source; 12 CPU workers, no GPU, no C09/C10/model/graph/planner.",
        },
        "estimated_cost": {"compute": "12-process Zarr decode/re-encode plus exact reread verification", "wall_time_hours": 6.0, "host_ram_gb": 24, "gpu": 0, "disk_gb": 20},
        "retention": "Keep the failed source run immutable and retain the corrected shards, exact raw-array digests, source/corrected tree hashes, manifests, logs, environment, RUN_STATE and seal.",
        "failure_policy": "Any non-resource source failure, logical byte drift, metadata drift, missing shard, source seal drift or corrected size over 20 GiB stops and seals FAIL. Never rerender, delete frames, relax the gate or overwrite either run.",
    }
    write(CARD, card)
    tools = {
        "runner": "tools/v3/run_primitive_relation_p1a_lossless_storage_corrective_v1.py",
        "freezer": "tools/v3/freeze_primitive_relation_p1a_lossless_storage_corrective_v1_spec.py",
        "repack": "src/mtare_topo/data/primitive_relation_lossless_repack.py",
        "repack_tests": "tests/v3/unit/test_primitive_relation_lossless_repack.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    inputs = [SOURCE / "artifacts/evidence_sha256.txt", SOURCE / "artifacts/task_manifest.json", SOURCE / "metrics/summary.json", SOURCE / "RUN_STATE.json"]
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "operation": "data_export", "date": "20260830",
        "slug": "primitive_relation_p1a_lossless_storage_corrective_v1", "seed": 0,
        "question": "Can the scientifically valid but resource-over-limit P1a corpus be represented under 20 GiB with zero logical data change?",
        "method": "Repack every sealed P1a Zarr array with zstd level 9 byte-shuffle while retaining chunks/schema and proving raw dtype-byte equality after every destination chunk is reopened.",
        "baseline": "The immutable zstd5 bitshuffle P1a run; all of its scientific/data checks passed and only its 20 GiB resource check failed.",
        "fallback": "Any logical drift or remaining resource excess stops the route; no rerendering, frame deletion, threshold change or in-place rewrite.",
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval, "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {"worlds": 80, "tasks": 240, "arrays_per_task": 11, "frames": 757290, "rays": 8723980800, "primitives": 24117, "paired_pose_corrections": 402, "unit_tests": 2, "workers": 12, "optimizer_steps": 0, "model_inference_frames": 0, "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0},
        "expected_evidence": ["240 losslessly repacked Zarr shards, 2,640 raw-array digests, source/corrected tree hashes, copied codebooks/constructions, source seal verification, logs/environment/RUN_STATE and SHA-256 seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_tools": {name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "frozen_inputs": {str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=GSE P1a lossless storage corrective", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=600s", "86400s",
            "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python", "tools/v3/run_primitive_relation_p1a_lossless_storage_corrective_v1.py",
            "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"), "--workers", "12",
        ],
    }
    write(SPEC, spec); print(SPEC)


if __name__ == "__main__":
    main()
