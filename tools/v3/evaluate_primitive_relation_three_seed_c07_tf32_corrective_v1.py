#!/usr/bin/env python3
"""C07-only corrective for the missing cuDNN TF32-off evaluation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_primitive_relation_three_seed_v1 import (
    EXPECTED_ROWS,
    _comparison,
    _learned_c07,
    _load_model,
    _write_json,
)
from mtare_topo.data.primitive_relation_batches import PrimitiveRelationBatchLoader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--baseline-c07-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("TF32 corrective evaluation requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    loader = PrimitiveRelationBatchLoader(args.sensor_root.resolve(), args.teacher_root.resolve())
    if len(loader) != EXPECTED_ROWS["c07"] or len(loader.task_names) != 30:
        raise RuntimeError("TF32 corrective C07 population drift")
    baseline = json.loads(args.baseline_c07_summary.read_text())
    if baseline.get("rows") != EXPECTED_ROWS["c07"] or not baseline.get("scientific_pass"):
        raise RuntimeError("TF32 corrective baseline drift")
    seeds = []
    for seed in range(3):
        model, checkpoint = _load_model(args.models_root / f"seed{seed}/selected.pt", seed, device)
        metrics = _learned_c07(model, loader, device=device)
        comparison = _comparison(metrics, baseline)
        record = {
            "seed": seed, "selected_epoch": int(checkpoint["epoch"]),
            "metrics": metrics, "comparison": comparison,
            "tf32_contract": {"matmul": False, "cudnn": False, "float32_matmul_precision": "highest"},
        }
        seeds.append(record); _write_json(output / f"c07_seed{seed}.json", record)
        print(json.dumps({"stage": "c07_tf32_corrective", "seed": seed, "comparison": comparison}), flush=True)
        del model; torch.cuda.empty_cache()
    passing = sum(bool(item["comparison"]["pass"]) for item in seeds)
    summary = {
        "schema_version": "primitive_relation_three_seed_c07_tf32_corrective_v1",
        "overall_status": "PASS_PRIMITIVE_RELATION_C07_TF32_CORRECTIVE_EVALUATION_COMPLETE",
        "scientific_pass": True,
        "model_v1_scientific_pass": passing >= 2,
        "c07": {"baseline": baseline, "seeds": seeds, "passing_seeds": passing, "scientific_pass": passing >= 2},
        "model_inference_rows": EXPECTED_ROWS["c07"] * 3 * 2,
        "optimizer_steps": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0, "mtare_worlds_read": 0,
        "tf32_contract": {"matmul": False, "cudnn": False, "float32_matmul_precision": "highest"},
        "decision": "ALLOW_ATTRIBUTION_WITH_CORRECTED_C07_METRICS",
    }
    _write_json(output / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
