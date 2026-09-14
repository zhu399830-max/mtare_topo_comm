"""Bounded local-token retrieval, not place/branch identity or registration.

All observed local tokens are retained by the producer. This retrieval view
uses a fixed coordinate-ordered subsample, never teacher IDs or model scores.
Zero vectors and observations without ROI support remain explicitly unknown.
"""
from dataclasses import dataclass
import re
import numpy as np


@dataclass(frozen=True)
class FrameBinding:
    source_key: str
    stamp_ns: int
    payload_sha256: str

    def __post_init__(self):
        if (not isinstance(self.source_key, str) or not self.source_key
                or type(self.stamp_ns) is not int or self.stamp_ns < 0
                or not re.fullmatch('[0-9a-f]{64}', self.payload_sha256)):
            raise ValueError('frame key, timestamp and payload SHA256 required')


@dataclass(frozen=True)
class LocalTokenRecord:
    record_id: str
    segment: str
    order: int
    source_refs: tuple[str, ...]
    xyz_current_sensor_m: np.ndarray
    tokens: np.ndarray
    source_frames: tuple[FrameBinding, ...]
    encoder_sha256: str
    token_model_sha256: str

    def __post_init__(self):
        if (not self.record_id or not self.segment or type(self.order) is not int or self.order < 0
                or not isinstance(self.source_refs, tuple) or not self.source_refs
                or any(not isinstance(s, str) or not s for s in self.source_refs)):
            raise ValueError('causal source-bound record required')
        if (not isinstance(self.source_frames, tuple) or not self.source_frames
                or any(not isinstance(f, FrameBinding) for f in self.source_frames)
                or tuple(f.source_key for f in self.source_frames) != self.source_refs
                or len(set(self.source_refs)) != len(self.source_refs)
                or any(a.stamp_ns >= b.stamp_ns for a, b in zip(self.source_frames, self.source_frames[1:]))
                or any(not re.fullmatch('[0-9a-f]{64}', h) for h in (self.encoder_sha256, self.token_model_sha256))):
            raise ValueError('ordered frame bindings and frozen model hashes required')
        xyz = np.array(self.xyz_current_sensor_m, dtype=np.float32, copy=True)
        tok = np.array(self.tokens, dtype=np.float32, copy=True)
        if (xyz.ndim != 2 or xyz.shape[1:] != (3,) or tok.shape != (len(xyz), 128)
                or not np.isfinite(xyz).all() or not np.isfinite(tok).all()):
            raise ValueError('finite per-token XYZ and 128D features required')
        # Bytes-backed arrays cannot be made writable with setflags(write=True).
        xyz = np.frombuffer(xyz.tobytes(), dtype=np.float32).reshape(xyz.shape)
        tok = np.frombuffer(tok.tobytes(), dtype=np.float32).reshape(tok.shape)
        object.__setattr__(self, 'xyz_current_sensor_m', xyz)
        object.__setattr__(self, 'tokens', tok)


def _retrieval_view(record):
    xyz, tokens = record.xyz_current_sensor_m, record.tokens
    ids = np.flatnonzero(np.linalg.norm(xyz, axis=1) <= 10.0)
    if not len(ids):
        return np.empty((0, 128), dtype=np.float32)
    # Tie on positions is broken by feature bytes; no order-dependent query ID.
    ids = sorted(ids.tolist(), key=lambda i: (tuple(xyz[i]), tokens[i].tobytes()))
    ids = np.asarray(ids)[np.linspace(0, len(ids) - 1, min(256, len(ids)), dtype=int)]
    chosen = tokens[ids]
    norms = np.linalg.norm(chosen, axis=1)
    if np.any(norms <= np.finfo(np.float32).eps):
        return np.empty((0, 128), dtype=np.float32)
    return chosen / norms[:, None]


def retrieve_structural_candidates(current, history):
    """Up to five past records, by symmetric local-token nearest similarity.

No single global mean vector; no acceptance threshold; no graph mutations.
Cross-segment candidates are intentionally excluded in this first interface.
"""
    if not isinstance(current, LocalTokenRecord):
        raise ValueError('typed current token record required')
    history = tuple(history)
    if (any(not isinstance(h, LocalTokenRecord) for h in history)
            or len({h.record_id for h in history}) != len(history)
            or any(h.order >= current.order or h.record_id == current.record_id
                   or h.source_frames[-1].stamp_ns >= current.source_frames[-1].stamp_ns for h in history)):
        raise ValueError('unique strictly past history required')
    if any((h.encoder_sha256, h.token_model_sha256) != (current.encoder_sha256, current.token_model_sha256) for h in history):
        raise ValueError('incompatible frozen representation versions')
    # The authenticated loader must verify payload hashes; this module checks
    # declaration consistency, it cannot prove an arbitrary caller is truthful.
    source_bindings = {}
    for record in (current, *history):
        for frame in record.source_frames:
            if frame.source_key in source_bindings and source_bindings[frame.source_key] != frame:
                raise ValueError('conflicting source payload or timestamp')
            source_bindings[frame.source_key] = frame
    query = _retrieval_view(current)
    rows, unknown = [], []
    for record in history:
        if record.segment != current.segment:
            unknown.append(dict(record_id=record.record_id, reason='different_segment'))
            continue
        target = _retrieval_view(record)
        if not len(query) or not len(target):
            unknown.append(dict(record_id=record.record_id, reason='no_nonzero_local_token_support'))
            continue
        sim = np.clip(query @ target.T, -1.0, 1.0)
        score = float(0.5 * (sim.max(axis=1).mean() + sim.max(axis=0).mean()))
        rows.append(dict(record_id=record.record_id, score=score,
                         current_source_refs=list(current.source_refs),
                         historical_source_refs=list(record.source_refs),
                         score_is_probability=False, association_verified=False))
    rows.sort(key=lambda row: (-row['score'], row['record_id']))
    return dict(candidates=rows[:5], eligible_records=len(rows), unknown=unknown,
                query_tokens=len(query), graph_mutated=False,
                registration_required=True)
