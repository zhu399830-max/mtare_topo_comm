"""Partial structural candidates, not physical openings or verified graph ports.

This interface preserves the trained 32 x 64 layout. Query indices are local
to this observation, never persistent node/port identities. Positions and
directions remain in current_sensor coordinates until an explicit transform.
"""
from dataclasses import dataclass
import math
import re


def _index(x, cap=None):
    if type(x) is not int or x<0 or (cap is not None and x>=cap):
        raise ValueError('bounded non-bool index required')


def _vector(x):
    if (type(x) is not tuple or len(x)!=3
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in x)):
        raise ValueError('immutable finite 3D vector required')


def _score(x):
    if type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1:
        raise ValueError('finite numerical confidence, not calibrated probability')


@dataclass(frozen=True)
class BranchDirectionCandidateV1:
    branch_query_index: int
    direction_current_sensor: tuple[float,float,float]
    confidence: float

    def __post_init__(self):
        _index(self.branch_query_index,64);_vector(self.direction_current_sensor);_score(self.confidence)
        if abs(math.hypot(*self.direction_current_sensor)-1)>1e-5:
            raise ValueError('unit direction required; no guessed fallback')


@dataclass(frozen=True)
class AnchorBranchCandidateV1:
    anchor_query_index: int
    position_current_sensor_m: tuple[float,float,float]
    confidence: float
    branches: tuple[BranchDirectionCandidateV1,...]

    def __post_init__(self):
        _index(self.anchor_query_index,32);_vector(self.position_current_sensor_m);_score(self.confidence)
        if math.hypot(*self.position_current_sensor_m)>10+1e-5:
            raise ValueError('anchor outside trained local domain')
        if (type(self.branches) is not tuple or len(self.branches)>64
                or any(type(b) is not BranchDirectionCandidateV1 for b in self.branches)):
            raise ValueError('at most64 typed directions per anchor; no truncation')
        if len({b.branch_query_index for b in self.branches})!=len(self.branches):
            raise ValueError('duplicate local branch query')


@dataclass(frozen=True)
class AnchorBranchObservationV1:
    stream_key: str
    decision_index: int
    source_frame_orders: tuple[int,int,int,int,int]
    input_binding_sha256: str
    anchors: tuple[AnchorBranchCandidateV1,...]

    @property
    def coordinate_frame(self):return 'current_sensor'

    @property
    def unavailable_fields(self):
        return ('opening_position','opening_width','opening_height','reachability',
                'calibrated_uncertainty','persistent_identity','verified_connection')

    def __post_init__(self):
        if type(self.stream_key) is not str or not self.stream_key.strip():raise ValueError('stream required')
        _index(self.decision_index)
        if type(self.source_frame_orders) is not tuple or len(self.source_frame_orders)!=5:
            raise ValueError('five causal source orders required')
        for i in self.source_frame_orders:_index(i)
        if any(a>=b for a,b in zip(self.source_frame_orders,self.source_frame_orders[1:])):
            raise ValueError('strict causal source order required')
        if type(self.input_binding_sha256) is not str or re.fullmatch('[0-9a-f]{64}',self.input_binding_sha256) is None:
            raise ValueError('input binding digest required')
        if (type(self.anchors) is not tuple or len(self.anchors)>32
                or any(type(a) is not AnchorBranchCandidateV1 for a in self.anchors)):
            raise ValueError('at most32 typed anchors required')
        if len({a.anchor_query_index for a in self.anchors})!=len(self.anchors):
            raise ValueError('duplicate local anchor query')
