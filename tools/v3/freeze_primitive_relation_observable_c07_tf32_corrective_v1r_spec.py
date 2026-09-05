#!/usr/bin/env python3
"""Correct only the exact-world trajectory inventory rejected by preflight."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_observable_c07_tf32_corrective_v1.json"
OLD_SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_observable_c07_tf32_corrective_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_observable_c07_tf32_corrective_v1r.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_observable_c07_tf32_corrective_v1r.json"
RUN_ID = "gate3_20260902_primitive_relation_observable_c07_tf32_corrective_v1r_seed0"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("observable C07 TF32 corrective V1R card/spec already exists")
    card = copy.deepcopy(json.loads(OLD_CARD.read_text()))
    spec = copy.deepcopy(json.loads(OLD_SPEC.read_text()))
    trajectories = [
        ("S01_flat_tree_small_C07", 1340, 1393.815973022, 696.907986511),
        ("S02_3d_tree_small_C07", 1252, 1308.988003860, 654.494001930),
        ("S03_flat_unicyclic_small_C07", 1266, 1321.499290027, 660.749645014),
        ("S04_3d_unicyclic_small_C07", 1518, 1575.764520061, 787.882260030),
        ("S05_flat_branch_medium_C07", 2620, 2709.480258383, 1354.740129192),
        ("S06_3d_branch_medium_C07", 2192, 2273.091660325, 1136.545830163),
        ("S07_flat_loop_rich_C07", 3404, 3530.803997363, 1765.401998681),
        ("S08_3d_loop_rich_C07", 4196, 4321.288770331, 2160.644385165),
        ("S09_flat_complex_C07", 4462, 4614.577572516, 2307.288786258),
        ("S10_3d_complex_C07", 5172, 5345.795578166, 2672.897789083),
    ]
    validation = [world for world, *_ in trajectories]
    card["card_id"] = "primitive_relation_observable_c07_tf32_corrective_v1r"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_OBSERVABLE_C07_TF32_CORRECTIVE_V1R"
    card["approval"]["scope"] = card["approval"]["scope"].replace("TF32-off corrective:", "TF32-off corrective V1R:")
    card["approval"]["confirmation_reference"] += " V1 preflight read zero data and rejected only the shorthand trajectory world; V1R enumerates the ten exact C07 worlds without changing the operation."
    card["worlds"]["validation"] = validation
    card["worlds"]["strict_test"] = [
        *[world.replace("_C07", "_C08") for world in validation],
        *[world.replace("_C07", "_C09") for world in validation],
        *[world.replace("_C07", "_C10") for world in validation],
        "all_M-TARE_benchmark_worlds",
    ]
    card["trajectories"] = [{
        "id": f"{world}:all_directed_traversals_same_checkpoint_corrective",
        "world": world, "split": "validation", "independent": True,
        "duration_s": duration, "distance_m": distance,
        "spatial_coverage_m": coverage,
    } for world, duration, distance, coverage in trajectories]
    write(CARD, card)

    spec["slug"] = "primitive_relation_observable_c07_tf32_corrective_v1r"
    spec["data_card"] = str(CARD.relative_to(PROJECT_ROOT))
    spec["config_path"] = str(CARD.relative_to(PROJECT_ROOT))
    spec["user_authorization"] = card["approval"]
    spec["corrective_of_preflight"] = str(OLD_SPEC.relative_to(PROJECT_ROOT))
    spec["method"] += " V1R changes only the Data Card trajectory inventory from one shorthand world to ten exact C07 parent-world records."
    spec["command"][-5] = "tools/v3/run_primitive_relation_observable_c07_tf32_corrective_v1r.py"
    spec["command"][-3] = str(SPEC)
    spec["command"][-1] = str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")
    inputs = {}
    for relative in spec["frozen_inputs"]:
        path = PROJECT_ROOT / relative
        if path == OLD_CARD:
            path = CARD
        inputs[str(path.relative_to(PROJECT_ROOT))] = sha(path)
    inputs[str(OLD_CARD.relative_to(PROJECT_ROOT))] = sha(OLD_CARD)
    inputs[str(OLD_SPEC.relative_to(PROJECT_ROOT))] = sha(OLD_SPEC)
    spec["frozen_inputs"] = inputs
    tools = {name: record["path"] for name, record in spec["frozen_tools"].items()}
    tools.update({
        "runner": "tools/v3/run_primitive_relation_observable_c07_tf32_corrective_v1r.py",
        "base_runner": "tools/v3/run_primitive_relation_observable_c07_tf32_corrective_v1.py",
        "freezer": "tools/v3/freeze_primitive_relation_observable_c07_tf32_corrective_v1r_spec.py",
    })
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": sha(PROJECT_ROOT / path)}
        for name, path in tools.items()
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
