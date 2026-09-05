#!/usr/bin/env python3
"""V9 ROS node with verified re-anchor and causal frontier-event feedback."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import semantic_topology_global_node_v3 as base
from mtare_topo.integration.online_topology_runtime_v3 import OnlineTopologyPlannerRuntimeV3


class SemanticTopologyGlobalNodeV5(base.SemanticTopologyGlobalNodeV3):
    """Reuse the frozen sensor/model interface and replace only graph runtime."""

    def __init__(self, **kwargs) -> None:
        original_runtime = base.OnlineTopologyPlannerRuntime
        base.OnlineTopologyPlannerRuntime = OnlineTopologyPlannerRuntimeV3
        try:
            super().__init__(**kwargs)
        finally:
            base.OnlineTopologyPlannerRuntime = original_runtime
        if not isinstance(self.runtime, OnlineTopologyPlannerRuntimeV3):
            raise RuntimeError("V5 failed to establish the combined corrective runtime")

    def write_snapshot(self) -> None:
        with self.lock:
            if self._snapshot_written:
                return
            snapshot = {
                "schema_version": "semantic_topology_global_node_v5_snapshot_v1",
                "mode": "shadow" if self.shadow else "closed_loop",
                "checkpoint": self.checkpoint_identity,
                "graph_config": base.GRAPH_CONFIG.to_dict(),
                "planner_config": base.PLANNER_CONFIG.to_dict(),
                "publish_period_sec": self.publish_period_sec,
                "input_operator": self.input_operator,
                "failed_cycles": self.failed_cycles,
                "input_points": self.input_points,
                "accepted_points": self.accepted_points,
                "unique_valid_cells": self.unique_valid_cells,
                "learned_empty_cycles": self.learned_empty_cycles,
                "fallback_cycles": self.fallback_cycles,
                "post_warmup_cycles": self.post_warmup_cycles,
                "post_warmup_fallback_cycles": self.post_warmup_fallback_cycles,
                "post_warmup_fallback_rate": self.post_warmup_fallback_cycles / max(self.post_warmup_cycles, 1),
                "wall_runtime_sec": time.monotonic() - self.started_monotonic,
                "runtime": self.runtime.snapshot(),
            }
            with (self.output / "topology_snapshot.json").open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n")
            if not self.trace.closed:
                self.trace.close()
            self._snapshot_written = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--publish-period-sec", type=float, default=1.0)
    args = parser.parse_args()
    import rospy

    rospy.init_node("semantic_topology_global_node_v5", anonymous=False)
    SemanticTopologyGlobalNodeV5(
        checkpoint=args.checkpoint.resolve(), checkpoint_sha256=args.checkpoint_sha256,
        output=args.output.resolve(), shadow=args.shadow,
        publish_period_sec=args.publish_period_sec,
    )
    rospy.loginfo("semantic topology v5 started in %s mode", "shadow" if args.shadow else "closed_loop")
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
