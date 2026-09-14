"""Pure-Python contract for handing global targets to retained M-TARE nodes."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from mtare_topo.planning.topological_frontier import GlobalTarget


@dataclass(frozen=True)
class MTAReHandoffOutput:
    waypoint: dict[str, Any] | None
    exploration_finish: bool
    runtime_sec: float
    map_clearing_radius_m: float | None
    planner_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MTAReGlobalHandoff:
    """Mirror the public high-level behavior required by the retained stack.

    ROS serialization is deliberately outside this class so contract tests run
    without ROS.  The wrapper must publish ``waypoint`` as PointStamped in the
    map frame and the other fields on the audited M-TARE topics.
    """

    def __init__(self) -> None:
        self._consecutive_empty_free_paths = 0

    def observe_free_path_count(self, count: int) -> float | None:
        if count < 0:
            raise ValueError("free path count must be non-negative")
        self._consecutive_empty_free_paths = self._consecutive_empty_free_paths + 1 if count == 0 else 0
        return 8.0 if self._consecutive_empty_free_paths >= 2 else None

    def package(
        self,
        target: GlobalTarget,
        *,
        stamp_sec: float,
        cycle_started_monotonic: float,
        map_clearing_radius_m: float | None = None,
        completion_confirmed: bool = False,
    ) -> MTAReHandoffOutput:
        if not math.isfinite(stamp_sec):
            raise ValueError("waypoint timestamp must be finite")
        runtime = time.monotonic() - cycle_started_monotonic
        if runtime < 0 or not math.isfinite(runtime):
            raise ValueError("planner runtime must be finite and non-negative")
        waypoint = None
        if target.status == "TARGET":
            if target.waypoint_xyz_m is None:
                raise ValueError("TARGET requires waypoint coordinates")
            waypoint = {
                "message_type": "geometry_msgs/PointStamped",
                "frame_id": "map",
                "stamp_sec": float(stamp_sec),
                "xyz_m": list(map(float, target.waypoint_xyz_m)),
            }
        if completion_confirmed and target.status != "CANDIDATE_COMPLETE":
            raise ValueError("completion can only confirm a candidate-complete planner state")
        finish = bool(completion_confirmed)
        return MTAReHandoffOutput(
            waypoint=waypoint,
            exploration_finish=finish,
            runtime_sec=float(runtime),
            map_clearing_radius_m=map_clearing_radius_m,
            planner_status=target.status,
        )
