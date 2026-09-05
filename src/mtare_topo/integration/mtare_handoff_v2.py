"""Corrected M-TARE handoff and causal local-rejection bookkeeping."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
from typing import Mapping

from mtare_topo.integration.mtare_handoff import MTAReGlobalHandoff, MTAReHandoffOutput
from mtare_topo.planning.topological_frontier import FrontierIdentity


class FrontierRetryLedger:
    """Count only completed local-planner rejection episodes per frontier."""

    def __init__(self) -> None:
        self._counts: defaultdict[FrontierIdentity, int] = defaultdict(int)

    def record_rejection(self, frontier: FrontierIdentity | None) -> None:
        if frontier is not None:
            self._counts[frontier] += 1

    def counts(self) -> Mapping[FrontierIdentity, int]:
        return dict(self._counts)

    def to_dict(self) -> dict[str, int]:
        return {
            f"{identity.node_id}:{identity.stub_index}": count
            for identity, count in sorted(
                self._counts.items(), key=lambda item: (item[0].node_id, item[0].stub_index)
            )
        }


class MTAReGlobalHandoffV2(MTAReGlobalHandoff):
    """Match the original TARE two-empty-path recovery episode exactly.

    V1 emitted 8 m clearing on every empty message after the second one.  The
    retained C++ implementation resets its counter after each publication, so
    a continuing empty stream produces one clearing event per pair.
    """

    def observe_free_path_count(self, count: int) -> float | None:
        if count < 0:
            raise ValueError("free path count must be non-negative")
        self._consecutive_empty_free_paths = self._consecutive_empty_free_paths + 1 if count == 0 else 0
        if self._consecutive_empty_free_paths >= 2:
            self._consecutive_empty_free_paths = 0
            return 8.0
        return None

    def package(self, target, **kwargs) -> MTAReHandoffOutput:
        """Publish an explicit hold waypoint for every non-TARGET state.

        Leaving the previous waypoint latched would let the retained local
        planner continue toward a stale branch during a semantic empty frame.
        All planner hold states already carry the current robot position.
        """

        output = super().package(target, **kwargs)
        if output.waypoint is not None:
            return output
        if target.waypoint_xyz_m is None or len(target.waypoint_xyz_m) != 3:
            raise ValueError("non-TARGET planner states require a hold waypoint")
        xyz = [float(value) for value in target.waypoint_xyz_m]
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("hold waypoint must be finite")
        return replace(output, waypoint={
            "message_type": "geometry_msgs/PointStamped",
            "frame_id": "map",
            "stamp_sec": float(kwargs["stamp_sec"]),
            "xyz_m": xyz,
            "hold": True,
        })


__all__ = ["FrontierRetryLedger", "MTAReGlobalHandoffV2", "MTAReHandoffOutput"]
