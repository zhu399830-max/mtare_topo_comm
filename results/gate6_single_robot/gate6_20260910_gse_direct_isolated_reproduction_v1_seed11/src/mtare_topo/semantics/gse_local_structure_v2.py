"""Output-only local structure contract; neither a detector nor a GT interface.

Probabilities are predictions, numerical support is not a detection, and a
binding digest is a caller assertion until verified against the source reader.
"""
from dataclasses import dataclass
import math
import re


def _index(value, limit=None):
    if type(value) is not int or value < 0 or (limit is not None and value >= limit):
        raise ValueError("nonnegative integer index within capacity required")


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("finite probability in [0,1] required")


def _vector(value, size):
    if type(value) is not tuple or len(value) != size or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value
    ):
        raise ValueError("immutable finite coordinate tuple required")


def _distribution(value, size):
    _vector(value, size)
    for v in value:
        _probability(v)
    if abs(sum(value) - 1.0) > 1e-5:
        raise ValueError("probability distribution must sum to one")


@dataclass(frozen=True)
class SourceFrameV2:
    frame_key: str
    order_index: int

    def __post_init__(self):
        if type(self.frame_key) is not str or not self.frame_key:
            raise ValueError("nonempty opaque source frame key required")
        _index(self.order_index)


@dataclass(frozen=True)
class StructureInstanceV2:
    query_index: int
    center_m: tuple[float, float, float]
    # corridor, junction, terminal, no_object; unknown is NOT no_object.
    event_probabilities: tuple[float, float, float, float]
    uncertainty: float
    numerically_supported: bool

    def __post_init__(self):
        _index(self.query_index, 32)
        _vector(self.center_m, 3)
        _distribution(self.event_probabilities, 4)
        _probability(self.uncertainty)
        if type(self.numerically_supported) is not bool:
            raise ValueError("explicit numerical support required")

    @property
    def existence_probability(self):
        return 1.0 - self.event_probabilities[3]


@dataclass(frozen=True)
class LocalOpeningV2:
    query_index: int
    position_m: tuple[float, float, float] | None
    direction: tuple[float, float, float] | None
    width_m: float | None
    height_m: float | None
    existence_probability: float
    # Distribution in the parent observation's structure-query order + unknown.
    structure_membership: tuple[float, ...]
    uncertainty: float
    numerically_supported: bool
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        _index(self.query_index, 64)
        if self.position_m is not None:
            _vector(self.position_m, 3)
        if self.direction is not None:
            _vector(self.direction, 3)
            if abs(math.hypot(*self.direction) - 1) > 1e-5:
                raise ValueError("known direction must be unit length")
        for v in (self.width_m, self.height_m):
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0):
                raise ValueError("dimension must be positive or explicitly unknown")
        if type(self.structure_membership) is not tuple or not 1 <= len(self.structure_membership) <= 33:
            raise ValueError("membership requires structures plus unknown")
        _distribution(self.structure_membership, len(self.structure_membership))
        for v in (self.existence_probability, self.uncertainty):
            _probability(v)
        if type(self.numerically_supported) is not bool:
            raise ValueError("explicit support required")
        if type(self.evidence_refs) is not tuple or any(type(x) is not str or not x for x in self.evidence_refs):
            raise ValueError("immutable nonempty evidence references required")


@dataclass(frozen=True)
class LocalStructureObservationV2:
    runtime_stream_key: str
    decision_index: int
    coordinate_frame: str
    source_frames: tuple[SourceFrameV2, ...]
    current_source_index: int
    input_binding_sha256: str
    structures: tuple[StructureInstanceV2, ...]
    openings: tuple[LocalOpeningV2, ...]
    # Do not interpret entropy or an uncalibrated head as a safety guarantee.
    uncertainty_calibrated: bool = False
    schema_version: str = "gse_local_structure_observation_v2"

    def __post_init__(self):
        if self.schema_version != "gse_local_structure_observation_v2":
            raise ValueError("unsupported schema version")
        for s in (self.runtime_stream_key, self.coordinate_frame):
            if type(s) is not str or not s:
                raise ValueError("explicit stream and coordinate frame required")
        _index(self.decision_index)
        _index(self.current_source_index)
        if type(self.source_frames) is not tuple or len(self.source_frames) != 5 or any(type(f) is not SourceFrameV2 for f in self.source_frames):
            raise ValueError("exactly five typed causal source frames required")
        orders = [f.order_index for f in self.source_frames]
        if any(a >= b for a, b in zip(orders, orders[1:])) or orders[-1] != self.current_source_index:
            raise ValueError("sources must be past-to-current, without future frames")
        if len({f.frame_key for f in self.source_frames}) != 5:
            raise ValueError("duplicate source frame identity")
        if type(self.input_binding_sha256) is not str or re.fullmatch(r"[a-f0-9]{64}", self.input_binding_sha256) is None:
            raise ValueError("source binding digest required")
        for values, cls, limit in ((self.structures, StructureInstanceV2, 32), (self.openings, LocalOpeningV2, 64)):
            if type(values) is not tuple or len(values) > limit or any(type(v) is not cls for v in values):
                raise ValueError("typed immutable output population exceeds capacity")
            if len({v.query_index for v in values}) != len(values):
                raise ValueError("duplicate query index")
        if any(len(p.structure_membership) != len(self.structures) + 1 for p in self.openings):
            raise ValueError("membership order must match structure population plus unknown")
        if type(self.uncertainty_calibrated) is not bool:
            raise ValueError("calibration state must be explicit")
