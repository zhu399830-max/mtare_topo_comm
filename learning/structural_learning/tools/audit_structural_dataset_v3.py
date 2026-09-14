from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return {"count": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))} if len(a) else {"count": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="results/structural_dataset_v3")
    args = parser.parse_args()
    root = Path(args.dataset_dir)
    by_robot, by_split, all_rows = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), []
    for split in ("train", "val"):
        for path in sorted((root / split).glob("*.npz")):
            with np.load(path, allow_pickle=False) as d:
                x = d["input_surface"]
                row = {"surface_cells": int((x[0] > 0.5).sum()), "mean_nonzero_density": float(x[1][x[0] > 0.5].mean()), "raw_support": float(d["geometry_metrics"][0]), "robot": str(d["robot"]), "split": split}
            all_rows.append(row)
            for key in ("surface_cells", "mean_nonzero_density", "raw_support"):
                by_robot[row["robot"]][key].append(row[key]); by_split[split][key].append(row[key])
    serial = lambda group: {name: {metric: describe(values) for metric, values in metrics.items()} for name, metrics in group.items()}
    output = {"purpose": "descriptive source-dependence audit only; no source classifier was trained", "sample_count": len(all_rows), "by_robot": serial(by_robot), "by_split": serial(by_split), "interpretation": "robot/split are metadata only. Surface-cell and density distributions are reported to expose, not hide, sensor/trajectory correlations."}
    out = root / "logs" / "source_dependence_audit.json"
    out.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
