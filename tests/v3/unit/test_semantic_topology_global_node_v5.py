from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "semantic_topology_global_node_v5",
        ROOT / "tools/v3/semantic_topology_global_node_v5.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_substitutes_combined_runtime_only_during_base_construction(monkeypatch):
    module = _load()
    base = sys.modules["semantic_topology_global_node_v3"]
    original = base.OnlineTopologyPlannerRuntime
    captured = {}

    def fake_init(self, **kwargs):
        captured["runtime_class"] = base.OnlineTopologyPlannerRuntime
        self.runtime = object.__new__(module.OnlineTopologyPlannerRuntimeV3)

    monkeypatch.setattr(base.SemanticTopologyGlobalNodeV3, "__init__", fake_init)
    node = module.SemanticTopologyGlobalNodeV5(checkpoint=Path("x"), checkpoint_sha256="x", output=Path("x"), shadow=False, publish_period_sec=1.0)
    assert captured["runtime_class"] is module.OnlineTopologyPlannerRuntimeV3
    assert isinstance(node.runtime, module.OnlineTopologyPlannerRuntimeV3)
    assert base.OnlineTopologyPlannerRuntime is original


def test_v5_source_keeps_frozen_v3_sensor_model_interface():
    source = (ROOT / "tools/v3/semantic_topology_global_node_v5.py").read_text()
    assert "semantic_topology_global_node_v3 as base" in source
    assert "OnlineTopologyPlannerRuntimeV3" in source
    assert "retry_penalty" not in source
    assert "direction_threshold" not in source
