"""Deployment metadata for a single causal five-frame structure observation.

This validates only the supplied metadata. The input producer remains responsible
for binding the actual point/feature tensors to these frames and for auditing
dataset splits. Passing this contract is not a dataset-wide causality proof.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class CausalFrameOrderContext:
    """Cached observation with genuine frame order but no measured clock.

    Sequence ID is provenance only; integer row numbers are NOT seconds.
    """
    coordinate_frame: str
    source_frame_ids: tuple[int, int, int, int, int]
    observation_frame_id: int
    sequence_id: str
    time_basis: str = 'frame_order_only'

    def __post_init__(self):
        _identifier(self.coordinate_frame, 'coordinate_frame')
        _identifier(self.sequence_id, 'sequence_id')
        ids = self.source_frame_ids
        if (type(ids) is not tuple or len(ids) != 5 or any(type(i) is not int or i < 0 for i in ids)
                or any(a >= b for a, b in zip(ids, ids[1:]))
                or type(self.observation_frame_id) is not int or self.observation_frame_id < ids[-1]
                or self.time_basis != 'frame_order_only'):
            raise ValueError('five causal ordered frame indices required; no fabricated seconds')


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or any(char.isspace() for char in value):
        raise ValueError(f"{name} must be a nonempty string without whitespace")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")


def _timestamp(value: float, name: str) -> float:
    # bool, strings, and array/tensor scalars are not accepted by coercion.
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite built-in int or float")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True, slots=True)
class CausalStructureContext:
    """One observation's source provenance, not a model feature or GT identity.

    Frame IDs are nonnegative sequence-local integer indices, supplied in temporal
    order; they need not be consecutive. Sequence identity is only provenance.
    Timestamps use one caller-declared clock. The observation can be published
    after its latest source frame, but none of its five sources may be future.
    """

    timestamp_s: float
    coordinate_frame: str
    source_frame_ids: tuple[int, int, int, int, int]
    source_timestamps_s: tuple[float, float, float, float, float]
    sequence_id: str

    def __post_init__(self) -> None:
        _identifier(self.coordinate_frame, "coordinate_frame")
        _identifier(self.sequence_id, "sequence_id")
        timestamp_s = _timestamp(self.timestamp_s, "timestamp_s")

        if type(self.source_frame_ids) is not tuple or len(self.source_frame_ids) != 5:
            raise ValueError("source_frame_ids must be a tuple of exactly five IDs")
        if any(type(value) is not int or value < 0 for value in self.source_frame_ids):
            raise ValueError("source_frame_ids must be nonnegative built-in integers")
        if any(a >= b for a, b in zip(self.source_frame_ids, self.source_frame_ids[1:])):
            raise ValueError("source_frame_ids must be unique and strictly increasing")

        if type(self.source_timestamps_s) is not tuple or len(self.source_timestamps_s) != 5:
            raise ValueError("source_timestamps_s must be a tuple of exactly five timestamps")
        times = tuple(_timestamp(value, "source_timestamps_s") for value in self.source_timestamps_s)
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("source_timestamps_s must be strictly increasing")
        if times[-1] > timestamp_s:
            raise ValueError("source_timestamps_s contains a future frame")

        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "source_timestamps_s", times)
