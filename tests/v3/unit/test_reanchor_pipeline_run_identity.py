from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mtare_topo.governance import build_run_id


ROOT = Path(__file__).resolve().parents[3]


def _load(name: str):
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/v3/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_material_reanchor_runner_ids_match_governance_builder() -> None:
    planned = (
        ("run_aee_composite_v9_reanchor_probe_v1", "aee_composite_v9_reanchor_probe_v1", 20260822),
        ("run_aee_composite_v9_reanchor_readiness_v1", "aee_composite_v9_reanchor_readiness_v1", 20260822),
        ("run_aee_composite_v9_reanchor_stochastic_v1", "aee_composite_v9_reanchor_stochastic_v1", 20260820),
    )
    for module_name, slug, seed in planned:
        module = _load(module_name)
        generated = build_run_id({
            "gate": 6,
            "date": "20260823",
            "slug": slug,
            "seed": seed,
        })
        assert module.RUN_ID == generated

