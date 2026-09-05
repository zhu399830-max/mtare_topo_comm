#!/usr/bin/env python3
"""V1R system corrective for the unavailable ``torch.flatnonzero`` API."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

from execute_gse_relational_exit_transport_readiness_v1 import main as v1_main


PASS_V1 = "PASS_GSE_RELATIONAL_EXIT_TRANSPORT_READINESS_V1"
PASS_V1R = "PASS_GSE_RELATIONAL_EXIT_TRANSPORT_READINESS_V1R"
FAIL_V1 = "FAIL_GSE_RELATIONAL_EXIT_TRANSPORT_READINESS_V1"
FAIL_V1R = "FAIL_GSE_RELATIONAL_EXIT_TRANSPORT_READINESS_V1R"


def flatnonzero_compat(value: torch.Tensor) -> torch.Tensor:
    """Exact PyTorch equivalent of NumPy ``flatnonzero`` for one vector."""

    if value.ndim != 1:
        raise ValueError("flatnonzero compatibility accepts one vector")
    return torch.nonzero(value, as_tuple=False).flatten()


def _output_dir(arguments: list[str]) -> Path:
    try:
        return Path(arguments[arguments.index("--output-dir") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("V1R output directory argument is missing") from exc


def main() -> int:
    if hasattr(torch, "flatnonzero"):
        raise RuntimeError("V1R compatibility is unnecessary in this Torch environment")
    torch.flatnonzero = flatnonzero_compat  # type: ignore[attr-defined]
    code = int(v1_main())
    output = _output_dir(sys.argv[1:])
    summary_path = output / "summary.json"
    source_path = output / "figure_source.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = "gse_relational_exit_transport_readiness_v1r"
    summary["status"] = PASS_V1R if summary.get("status") == PASS_V1 else FAIL_V1R
    summary["system_corrective"] = "torch.flatnonzero replaced by torch.nonzero(..., as_tuple=False).flatten(); no model/data/check/threshold change"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["schema_version"] = "gse_relational_exit_transport_readiness_figure_source_v1r"
    source["summary"] = summary
    source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
