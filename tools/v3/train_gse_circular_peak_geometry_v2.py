#!/usr/bin/env python3
"""Train V1 architecture with the frozen V2 peak-presence corrective only."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

import train_gse_circular_peak_geometry_v1 as v1
from mtare_topo.representation.gse_soft_angular_peak_loss import (
    FOCAL_ALPHA,
    FOCAL_BETA,
    SUPPORT_RADIUS_BINS,
    replace_v1_presence_with_v2,
)


_V1_LOSS = v1.circular_peak_geometry_loss


def _v2_loss(outputs, targets):
    return replace_v1_presence_with_v2(
        _V1_LOSS(outputs, targets),
        outputs["peak_presence_logits"],
        targets["presence"],
    )


def _argument_value(name: str) -> Path:
    index = sys.argv.index(name)
    return Path(sys.argv[index + 1]).resolve()


def main() -> int:
    output_dir = _argument_value("--output-dir")
    v1.circular_peak_geometry_loss = _v2_loss
    return_code = v1.main()
    checkpoint_path = output_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["schema_version"] = "gse_circular_peak_geometry_checkpoint_v2"
    checkpoint["peak_presence_loss"] = {
        "name": "soft_angular_focal_plus_equal_count_top_hard_negative_ranking",
        "support_radius_bins": SUPPORT_RADIUS_BINS,
        "focal_alpha": FOCAL_ALPHA,
        "focal_beta": FOCAL_BETA,
    }
    torch.save(checkpoint, checkpoint_path)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = "gse_circular_peak_geometry_training_seed_v2"
    summary["peak_presence_loss"] = checkpoint["peak_presence_loss"]
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
