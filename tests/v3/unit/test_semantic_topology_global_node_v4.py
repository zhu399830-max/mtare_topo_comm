from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from mtare_topo.integration.online_topology_runtime_v2 import OnlineTopologyPlannerRuntimeV2


ROOT = Path(__file__).resolve().parents[3]


def load_node_module():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "semantic_topology_global_node_v4",
        ROOT / "tools/v3/semantic_topology_global_node_v4.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_constructs_base_node_with_reanchor_runtime_then_restores_global(monkeypatch) -> None:
    module = load_node_module()
    observed = []

    def fake_init(self, **kwargs):
        observed.append(module.base.OnlineTopologyPlannerRuntime)
        self.runtime = object()

    monkeypatch.setattr(module.base.SemanticTopologyGlobalNodeV3, "__init__", fake_init)
    original = module.base.OnlineTopologyPlannerRuntime
    try:
        # The final identity check deliberately rejects the fake runtime; the
        # observation still proves the base constructor saw the V2 class.
        try:
            module.SemanticTopologyGlobalNodeV4(checkpoint=Path("unused"))
        except RuntimeError as exc:
            assert "failed to establish" in str(exc)
    finally:
        assert module.base.OnlineTopologyPlannerRuntime is original
    assert observed == [OnlineTopologyPlannerRuntimeV2]


def test_v4_has_distinct_ros_and_snapshot_identity() -> None:
    source = (ROOT / "tools/v3/semantic_topology_global_node_v4.py").read_text(encoding="utf-8")
    assert 'rospy.init_node("semantic_topology_global_node_v4"' in source
    assert '"semantic_topology_global_node_v4_snapshot_v1"' in source
    assert "OnlineTopologyPlannerRuntimeV2" in source
