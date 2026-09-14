import argparse
import glob
from pathlib import Path

import numpy as np

from learning.local_structural_map.tools.io_utils import load_map_npz, write_json


def audit(paths: list[Path]) -> dict:
    if not paths:
        return {"count": 0}
    rows = []
    total = {
        "height_valid": 0,
        "occupied": 0,
        "intersection": 0,
        "height_valid_nonoccupied": 0,
        "occupied_height_invalid": 0,
        "cells": 0,
    }
    for path in paths:
        data = load_map_npz(path)
        names = data["channel_names"]
        tensor = data["tensor"]
        occupied = tensor[names.index("occupied_mask")] > 0.5
        if "endpoint_height_range" in names:
            height_valid = tensor[names.index("endpoint_height_range")] > 0
            for band in [n for n in names if n.startswith("occupancy_z_band_")]:
                height_valid |= tensor[names.index(band)] > 0.5
        else:
            height_names = [n for n in ["ground_height", "max_height_above_ground", "height_range"] if n in names]
            height_valid = np.zeros_like(occupied)
            for name in height_names:
                height_valid |= tensor[names.index(name)] > 0
        inter = occupied & height_valid
        union = occupied | height_valid
        rec = {
            "file": str(path),
            "height_valid_cells": int(height_valid.sum()),
            "occupied_cells": int(occupied.sum()),
            "intersection_cells": int(inter.sum()),
            "height_valid_nonoccupied_cells": int((height_valid & ~occupied).sum()),
            "occupied_height_invalid_cells": int((occupied & ~height_valid).sum()),
            "iou": float(inter.sum() / max(1, union.sum())),
        }
        rows.append(rec)
        total["height_valid"] += rec["height_valid_cells"]
        total["occupied"] += rec["occupied_cells"]
        total["intersection"] += rec["intersection_cells"]
        total["height_valid_nonoccupied"] += rec["height_valid_nonoccupied_cells"]
        total["occupied_height_invalid"] += rec["occupied_height_invalid_cells"]
        total["cells"] += occupied.size
    union_total = total["height_valid"] + total["occupied"] - total["intersection"]
    return {
        "count": len(paths),
        "summary": {
            **total,
            "iou": float(total["intersection"] / max(1, union_total)),
            "height_valid_ratio": float(total["height_valid"] / max(1, total["cells"])),
            "occupied_ratio": float(total["occupied"] / max(1, total["cells"])),
        },
        "per_file": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-glob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(p) for p in sorted(glob.glob(args.sample_glob))]
    write_json(args.output, audit(paths))


if __name__ == "__main__":
    main()
