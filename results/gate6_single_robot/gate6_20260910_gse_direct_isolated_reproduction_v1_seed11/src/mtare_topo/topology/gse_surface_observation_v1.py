"""Class-free output DTOs, not a detector or authenticated deployment binding.

Coordinates are in the current robot frame. Numerical support is not calibrated
existence or safety; UNKNOWN geometry stays None. Membership is independent per
anchor query, never a softmax assignment to one anchor. No teacher IDs belong
in this interface. Opaque source keys/hashes are caller assertions until bound
to an actual source reader and the exact model forward.
"""
from dataclasses import dataclass
import math
import re

from mtare_topo.semantics.gse_local_structure_v2 import SourceFrameV2


def _index(value, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value >= maximum):
        raise ValueError("non-bool integer index within capacity required")


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite non-bool number required")


def _probability(value):
    _number(value)
    if not 0 <= value <= 1:
        raise ValueError("probability outside [0,1]")


def _vector(value, size):
    if type(value) is not tuple or len(value) != size:
        raise ValueError("immutable fixed-size coordinate tuple required")
    for x in value:
        _number(x)


@dataclass(frozen=True)
class SurfaceAnchorV1:
    query_index: int
    position_robot_m: tuple[float, float, float] | None
    existence_probability: float
    uncertainty_m: tuple[float, float, float] | None
    numerically_supported: bool

    def __post_init__(self):
        _index(self.query_index, 32)
        _probability(self.existence_probability)
        if type(self.numerically_supported) is not bool:
            raise ValueError("explicit numerical support required")
        if self.position_robot_m is not None:
            _vector(self.position_robot_m, 3)
        if self.uncertainty_m is not None:
            _vector(self.uncertainty_m, 3)
            if min(self.uncertainty_m) < 0:
                raise ValueError("position uncertainty is nonnegative metres, not a class probability")
        if not self.numerically_supported and (self.position_robot_m is not None or self.uncertainty_m is not None):
            raise ValueError("unsupported anchor geometry must be explicitly unknown")


@dataclass(frozen=True)
class SurfaceOpeningV1:
    query_index: int
    position_robot_m: tuple[float, float, float] | None
    direction_robot: tuple[float, float, float] | None
    width_m: float | None
    height_m: float | None
    existence_probability: float
    numerically_supported: bool
    reachability_probabilities: tuple[float, float, float]  # reachable, blocked, unknown
    traversability: str  # explicit caller readout, not an independent safety certificate
    anchor_relation_probability: tuple[float, ...]  # query-index addressed, exactly32
    anchor_relation_valid: tuple[bool, ...]

    def __post_init__(self):
        _index(self.query_index, 64)
        _probability(self.existence_probability)
        if type(self.numerically_supported) is not bool:
            raise ValueError("explicit opening numerical support required")
        for xyz in (self.position_robot_m, self.direction_robot):
            if xyz is not None:
                _vector(xyz, 3)
        if self.direction_robot is not None and abs(math.hypot(*self.direction_robot) - 1.) > 1e-5:
            raise ValueError("known opening direction must be unit3D")
        for size in (self.width_m, self.height_m):
            if size is not None:
                _number(size)
                if size <= 0:
                    raise ValueError("known opening dimensions must be positive")
        _vector(self.reachability_probabilities, 3)
        for p in self.reachability_probabilities:
            _probability(p)
        if abs(sum(self.reachability_probabilities) - 1.) > 1e-5:
            raise ValueError("reachability distribution must sum to one")
        if self.traversability not in ("traversable", "blocked", "unknown"):
            raise ValueError("explicit three-state traversability required")
        if self.traversability != "unknown":
            index = ("traversable", "blocked", "unknown").index(self.traversability)
            if any(self.reachability_probabilities[index] <= p for j, p in enumerate(self.reachability_probabilities) if j != index):
                raise ValueError("non-unknown readout must agree with unique maximum probability; not execution evidence")
        _vector(self.anchor_relation_probability, 32)
        for p in self.anchor_relation_probability:
            _probability(p)
        if (type(self.anchor_relation_valid) is not tuple or len(self.anchor_relation_valid) != 32
                or any(type(v) is not bool for v in self.anchor_relation_valid)):
            raise ValueError("32 independent boolean relation-valid masks required")
        if not self.numerically_supported and (
                any(v is not None for v in (self.position_robot_m, self.direction_robot, self.width_m, self.height_m))
                or any(self.anchor_relation_valid) or self.traversability != "unknown"):
            raise ValueError("unsupported openings cannot invent geometry, relation or traversability")


@dataclass(frozen=True)
class SurfaceObservationV1:
    runtime_stream_key: str
    decision_index: int
    source_frames: tuple[SourceFrameV2, ...]
    current_source_index: int
    input_binding_sha256: str
    anchors: tuple[SurfaceAnchorV1, ...]
    openings: tuple[SurfaceOpeningV1, ...]
    timestamp_s: float | None = None
    coordinate_frame: str = "robot"
    schema_version: str = "gse_surface_observation_v1"

    def __post_init__(self):
        if self.schema_version != "gse_surface_observation_v1" or self.coordinate_frame != "robot":
            raise ValueError("versioned robot-frame observation required")
        if type(self.runtime_stream_key) is not str or not self.runtime_stream_key.strip():
            raise ValueError("runtime stream key required")
        _index(self.decision_index); _index(self.current_source_index)
        if self.timestamp_s is not None:
            _number(self.timestamp_s)
        if (type(self.source_frames) is not tuple or len(self.source_frames) != 5
                or any(type(f) is not SourceFrameV2 for f in self.source_frames)):
            raise ValueError("exactly five typed causal source frames required")
        orders = [f.order_index for f in self.source_frames]
        if any(a >= b for a, b in zip(orders, orders[1:])) or orders[-1] != self.current_source_index:
            raise ValueError("five source frames must end at current, in strict past-to-current order")
        if len({f.frame_key for f in self.source_frames}) != 5:
            raise ValueError("duplicate source frame identity")
        if type(self.input_binding_sha256) is not str or re.fullmatch(r"[a-f0-9]{64}", self.input_binding_sha256) is None:
            raise ValueError("source/forward binding digest required")
        for values, cls, capacity in ((self.anchors, SurfaceAnchorV1, 32), (self.openings, SurfaceOpeningV1, 64)):
            if type(values) is not tuple or len(values) > capacity or any(type(v) is not cls for v in values):
                raise ValueError("immutable typed32anchors/64openings capacity contract required")
            if len({v.query_index for v in values}) != len(values):
                raise ValueError("duplicate observation-local query index")
        present = {a.query_index for a in self.anchors}
        if any(valid and i not in present for opening in self.openings
               for i, valid in enumerate(opening.anchor_relation_valid)):
            raise ValueError("known relation must reference a present anchor query")
