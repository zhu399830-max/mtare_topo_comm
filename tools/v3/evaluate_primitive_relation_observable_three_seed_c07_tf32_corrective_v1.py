#!/usr/bin/env python3
"""TF32-off corrective wrapper for the frozen observable C07 evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

import evaluate_primitive_relation_observable_three_seed_v1 as base


SCHEMA = "primitive_relation_observable_three_seed_c07_tf32_corrective_v1"


def configure_numerical_contract() -> dict[str, object]:
    """Match the deterministic contract used for training and epoch selection."""

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    contract = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    expected = {
        "deterministic_algorithms": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    if contract != expected:
        raise RuntimeError(f"observable corrective numerical contract drift: {contract}")
    return contract


def _output_directory(arguments: list[str]) -> Path:
    try:
        index = arguments.index("--output-dir")
        return Path(arguments[index + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("observable corrective output argument missing") from exc


def main() -> int:
    contract = configure_numerical_contract()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    output = _output_directory(sys.argv[1:])
    returncode = base.main()
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("observable corrective base summary missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["base_schema_version"] = summary.get("schema_version")
    summary["schema_version"] = SCHEMA
    summary["numerical_contract"] = contract
    summary["peak_cuda_allocated_bytes"] = int(
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    summary["peak_cuda_reserved_bytes"] = int(
        torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    )
    summary["corrective_scope"] = {
        "same_frozen_checkpoints": True,
        "same_c07_population_and_gates": True,
        "optimizer_steps": 0,
        "c08_rows_read": 0,
        "graph_replays": 0,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "stage": "tf32_corrective_contract",
        "numerical_contract": contract,
        "scientific_pass": summary.get("scientific_pass"),
        "decision": summary.get("decision"),
    }, sort_keys=True), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
