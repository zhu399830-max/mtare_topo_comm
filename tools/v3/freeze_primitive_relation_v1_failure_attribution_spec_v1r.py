#!/usr/bin/env python3
"""Freeze the Data-Card-only V1R corrective without changing science."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_v1_failure_attribution_v1.json"
OLD_SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_v1_failure_attribution_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_v1_failure_attribution_v1r.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_v1_failure_attribution_v1r.json"
RUN_ID = "gate3_20260831_primitive_relation_v1_failure_attribution_v1r_seed0"


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
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("primitive relation attribution V1R card/spec already exists")
    old_card = json.loads(OLD_CARD.read_text())
    old_spec = json.loads(OLD_SPEC.read_text())
    card = copy.deepcopy(old_card)
    card["card_id"] = "primitive_relation_v1_failure_attribution_v1r"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_V1_FAILURE_ATTRIBUTION_V1R"
    card["worlds"]["train"] = ["NONE_FROZEN_CHECKPOINT_AUDIT_ONLY"]
    card["purpose"] += " V1R changes only the required nonempty train-field representation after V1 preflight rejected an empty audit field."
    card["approval"]["scope"] = (
        "One immutable full-population C07-only frozen-checkpoint inference attribution: 10 parents, "
        "30 paired geometry tasks, 64644 sequences and seeds 0/1/2; zero optimizer, threshold change, "
        "C08/C09/C10, graph or M-TARE. V1R only represents the absent train population with an explicit NONE sentinel."
    )
    write(CARD, card)

    spec = copy.deepcopy(old_spec)
    spec["slug"] = "primitive_relation_v1_failure_attribution_v1r"
    spec["data_card"] = str(CARD.relative_to(PROJECT_ROOT))
    spec["config_path"] = str(CARD.relative_to(PROJECT_ROOT))
    spec["method"] += " V1R changes no science and only supplies the governance-required explicit NONE train sentinel."
    spec["user_authorization"] = card["approval"]
    spec["command"][-5] = "tools/v3/run_primitive_relation_v1_failure_attribution_v1r.py"
    spec["command"][-3] = str(SPEC)
    spec["command"][-1] = str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")

    frozen_inputs = {}
    for relative in old_spec["frozen_inputs"]:
        path = PROJECT_ROOT / relative
        if path == OLD_CARD:
            path = CARD
        frozen_inputs[str(path.relative_to(PROJECT_ROOT))] = sha(path)
    spec["frozen_inputs"] = frozen_inputs
    tools = {name: record["path"] for name, record in old_spec["frozen_tools"].items()}
    tools["runner"] = "tools/v3/run_primitive_relation_v1_failure_attribution_v1r.py"
    tools["base_runner"] = "tools/v3/run_primitive_relation_v1_failure_attribution.py"
    tools["freezer"] = "tools/v3/freeze_primitive_relation_v1_failure_attribution_spec_v1r.py"
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()
    }
    write(SPEC, spec); print(SPEC)


if __name__ == "__main__":
    main()
