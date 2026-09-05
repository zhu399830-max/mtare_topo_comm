#!/usr/bin/env python3
"""Freeze the identity-alignment corrective V2 Data Card and run spec."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_factorized_causal_node_c09_v2_seed0"
V1_RUN = "results/gate3_semantics/gate3_20260828_gse_factorized_causal_node_c09_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_causal_node_c09_v2.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_factorized_causal_node_c09_v2.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_causal_node_c09_v1.json"
SOURCE_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_factorized_causal_node_c09_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    card = deepcopy(load_json(SOURCE_CARD))
    card.update({
        "card_id": "gse_factorized_causal_node_c09_v2",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_CAUSAL_NODE_C09_V2",
        "purpose": "Repeat the frozen C09 causal decision-node qualification once after correcting only archive-row identity alignment.",
    })
    card["approval"] = {
        **card["approval"],
        "scope": "One immutable V2 corrective. The only change is explicit global_sequence_index/parent_id bijective reordering of already frozen baseline logits; data, models, threshold, metrics and gates are unchanged.",
    }
    card["source"]["failed_v1_run"] = V1_RUN
    card["split"]["historical_pollution_audit"] += (
        " V1 stopped before episode inference/evaluation because validation_outputs.npz uses a deterministic non-sorted row order."
    )
    card["methods"]["main"] += (
        " Frozen baseline logits are first reordered by the unique global_sequence_index and parent_id bijection; archive row position is forbidden as identity."
    )
    card["failure_policy"] += " A second implementation correction or V3 retry is forbidden."
    card["evidence"]["failure_policy"] = card["failure_policy"]
    write_json(CARD, card)

    spec = deepcopy(load_json(SOURCE_SPEC))
    spec.update({
        "slug": "gse_factorized_causal_node_c09_v2",
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": deepcopy(card["approval"]),
        "method": card["methods"]["main"],
        "fallback": card["methods"]["fallback"],
    })
    frozen_inputs = dict(spec["frozen_inputs"])
    for relative in (
        f"{V1_RUN}/RUN_STATE.json",
        f"{V1_RUN}/metrics/summary.json",
        f"{V1_RUN}/artifacts/evidence_sha256.txt",
        f"{V1_RUN}/logs/00_teacher_free_c09_inference.log",
    ):
        frozen_inputs[relative] = _sha(PROJECT_ROOT / relative)
    spec["frozen_inputs"] = frozen_inputs
    tool_paths = {
        name: record["path"] for name, record in spec["frozen_tools"].items()
        if name not in {"data_card", "freezer", "runner"}
    }
    tool_paths.update({
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "causal_runtime_alignment": "src/mtare_topo/representation/gse_causal_episode_runtime.py",
        "runtime_alignment_tests": "tests/v3/unit/test_gse_causal_episode_runtime.py",
        "runner_base": "tools/v3/run_gse_factorized_causal_node_c09_v1.py",
        "runner": "tools/v3/run_gse_factorized_causal_node_c09_v2.py",
        "freezer": "tools/v3/freeze_gse_factorized_causal_node_c09_spec_v2.py",
    })
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)}
        for name, path in tool_paths.items()
    }
    spec["command"] = [
        "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "3800s", PYTHON,
        "tools/v3/run_gse_factorized_causal_node_c09_v2.py",
        "--spec", str(SPEC), "--run-dir",
        str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
    ]
    spec["expected_evidence"] = [card["retention"]]
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT))
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
