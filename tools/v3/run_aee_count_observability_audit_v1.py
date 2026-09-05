#!/usr/bin/env python3
"""Audit whether fixed AEE masking removes teacher-exit evidence from Cano validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.aee_corrective_training import CorrectiveAEEMultitaskDataset, CorrectiveCanoMultitaskDataset, MaskMatchedCanoValidationDataset
from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.data.cano_sensor_smoke import ELEVATION_DEG
from mtare_topo.governance import load_json, write_json
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline, circular_distance_deg
import run_aee_head_adaptation_v1 as evidence


RUN_ID = "gate2_20260822_aee_count_observability_audit_v1_seed20260822"
STATUS = "COMPLETED_AEE_COUNT_OBSERVABILITY_AUDIT_V1"
MATCH_TOLERANCE_DEG = 20.0


def matched_flags(teacher: tuple[float, ...], predicted: list[float], tolerance_deg: float = MATCH_TOLERANCE_DEG) -> list[bool]:
    return [any(circular_distance_deg(value, candidate) <= tolerance_deg for candidate in predicted) for value in teacher]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("count-observability run identity/state mismatch")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    if spec.get("gate") != 2 or spec.get("operation") != "audit" or card.get("status") != "APPROVED_FOR_AEE_COUNT_OBSERVABILITY_AUDIT_V1":
        raise RuntimeError("approved Gate-2 audit scope required")
    for name, item in spec["frozen_tools"].items():
        if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen tool drift: {name}")
    source = (PROJECT_ROOT / spec["source_dataset_run"]).resolve()
    entries = evidence.verify_seal(source, spec["source_dataset_seal_sha256"])
    write_json(run_dir / "config/input_integrity.json", {"source_dataset_run":str(source.relative_to(PROJECT_ROOT)),"source_dataset_seal_sha256":spec["source_dataset_seal_sha256"],"source_seal_entries":entries,"validation_mask_seed":int(spec["validation_mask_seed"]),"match_tolerance_deg":MATCH_TOLERANCE_DEG,"c09_frames_read":0,"c10_frames_read":0})
    write_json(run_dir / "RUN_STATE.json", {"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING"})
    cano = CorrectiveCanoMultitaskDataset(source, "validation")
    aee = CorrectiveAEEMultitaskDataset(source)
    sparse = MaskMatchedCanoValidationDataset(cano, aee, int(spec["validation_mask_seed"]))
    baseline = RangeExitBaseline()
    rows = []
    lost_by_count: Counter[int] = Counter()
    teacher_exits = dense_visible = sparse_visible = lost_exits = frames_with_loss = 0
    dense_count_exact = sparse_count_exact = 0
    for index in range(5000):
        dense_item = cano[index]
        sparse_item = sparse[index]
        teacher = tuple(float(value) % 360.0 for value in dense_item["headings_robot_deg"])
        dense_output = baseline.predict(dense_item["student"][0] * 50.0, dense_item["student"][1] > 0.5, ELEVATION_DEG)
        sparse_output = baseline.predict(sparse_item["student"][0] * 50.0, sparse_item["student"][1] > 0.5, ELEVATION_DEG)
        dense_flags = matched_flags(teacher, dense_output["headings_robot_deg"])
        sparse_flags = matched_flags(teacher, sparse_output["headings_robot_deg"])
        lost = [dense and not masked for dense, masked in zip(dense_flags, sparse_flags, strict=True)]
        count = len(teacher)
        teacher_exits += count
        dense_visible += sum(dense_flags)
        sparse_visible += sum(dense and masked for dense, masked in zip(dense_flags, sparse_flags, strict=True))
        lost_exits += sum(lost)
        if any(lost):
            frames_with_loss += 1
            lost_by_count[count] += 1
        dense_count_exact += int(len(dense_output["headings_robot_deg"]) == count)
        sparse_count_exact += int(len(sparse_output["headings_robot_deg"]) == count)
        rows.append({
            "frame_id": dense_item["frame_id"], "teacher_count": count,
            "teacher_headings_deg": list(teacher), "dense_b0_headings_deg": dense_output["headings_robot_deg"],
            "sparse_b0_headings_deg": sparse_output["headings_robot_deg"], "dense_teacher_visible": dense_flags,
            "sparse_teacher_visible": sparse_flags, "mask_causally_lost_dense_visible": lost,
            "mask_source_frame_id": sparse_item["mask_source_frame_id"],
        })
    if len(rows) != 5000 or teacher_exits <= 0 or dense_visible <= 0:
        raise RuntimeError("count-observability audit denominator failed")
    (run_dir / "artifacts").mkdir(exist_ok=True)
    with (run_dir / "artifacts/observability.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema_version":"aee_count_observability_audit_v1", "overall_status":STATUS,
        "frames":5000, "teacher_exits":teacher_exits, "dense_b0_teacher_visible_exits":dense_visible,
        "sparse_retained_dense_visible_exits":sparse_visible, "mask_causally_lost_dense_visible_exits":lost_exits,
        "frames_with_mask_causal_exit_loss":frames_with_loss, "frames_with_loss_by_teacher_count":dict(sorted(lost_by_count.items())),
        "dense_b0_count_exact_frames":dense_count_exact, "sparse_b0_count_exact_frames":sparse_count_exact,
        "match_tolerance_deg":MATCH_TOLERANCE_DEG, "baseline_config":baseline.config.to_dict(),
        "interpretation":"MISMATCH_PRESENT" if lost_exits else "NO_MASK_CAUSAL_MISMATCH_DETECTED",
        "source_seal_entries":entries, "training_steps":0, "model_inference_frames":0,
        "c09_frames_read":0, "c10_frames_read":0, "formal_benchmark_frames_read":0,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    (run_dir / "logs/audit.log").write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    write_json(run_dir / "RUN_STATE.json", {"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED","overall_status":STATUS})
    sealed = evidence.seal(run_dir)
    print(json.dumps({**summary,"sealed_files":sealed}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
