"""Header/source-conservation audit and exact raw-window lookup, never ICP.

No bag I/O or point conversion. The producer must authenticate the pinned image,
source files and complete message-header sequences before calling this module.
No nearest-time fallback and no source-key renaming on cached model records.
The v1 source contract is specific to one uninterrupted AEE startup: six input
callbacks are discarded, then each callback emits at most one ordered output.
Reaching the output-count upper bound is useful only if the entire common
capture boundary is authenticated externally. This module verifies the supplied
header prefix, not that no further producer input existed outside the bag tail.
Its mapping remains conditional and must not enable native advice on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re


IMAGE = 'sha256:5cbede29e4229e92d85ebaf919be2fb9ebc4ad9ed97029b7639997779062748c'
SOURCE_SHA = {
    'vehicleSimulator.cpp': 'ed906e39762379074c2fbc4ac8a597403181315626d993fcfa43b6eb4f3c028e',
    'sensorScanGeneration.cpp': '90fd93740614e51d12c5cbf0a6caa0b384320b59de9b5a06d15b4a45037210b2',
    'GazeboRosVelodyneLaser.cpp': 'c4b022b7e578e4b1908933f5918984e3044023ffa2946ce84f1127c403ec4e96',
    'vehicle_simulator_seeded.launch': 'c9888a8e908cdd26971f6fdf541644682132143d4ad7e3a090677adf3b25498c',
}


@dataclass(frozen=True)
class ScanHeader:
    seq: int
    stamp_ns: int
    arrival_ns: int
    caller: str

    def __post_init__(self):
        if (any(type(x) is not int or x < 0 for x in (self.seq, self.stamp_ns, self.arrival_ns))
                or type(self.caller) is not str or not self.caller):
            raise ValueError('invalid scan header')


@dataclass(frozen=True)
class AuditedScanProvenance:
    segment_id: str
    bag_sha256: str
    raw: tuple[ScanHeader, ...]
    registered: tuple[ScanHeader, ...]
    image: str
    source_sha256: tuple[tuple[str, str], ...]
    scope: str = field(default='conditional_header_mapping_requires_common_capture_boundary', init=False)
    capture_boundary_verified: bool = field(default=False, init=False)

    def __post_init__(self):
        if (type(self.raw) is not tuple or type(self.registered) is not tuple or
                type(self.source_sha256) is not tuple or self.source_sha256 != tuple(sorted(SOURCE_SHA.items()))):
            raise ValueError('immutable complete source evidence required')
        _audit_fields(self.raw, self.registered, self.segment_id, self.bag_sha256,
                      self.image, dict(self.source_sha256))

    @property
    def bindings(self):
        return tuple((f'registered_scan:{g.stamp_ns}', f'/velodyne_points:{r.stamp_ns}', r.stamp_ns)
                     for r, g in zip(self.raw[6:], self.registered))

    @property
    def raw_count(self):
        return len(self.raw)


@dataclass(frozen=True)
class ModelWindowSource:
    window_id: str
    segment_id: str
    bag_sha256: str
    input_sha256: str
    raw_source_keys: tuple[str, ...]

    def __post_init__(self):
        if (any(type(x) is not str or not x for x in (self.window_id, self.segment_id)) or
                any(type(h) is not str or not re.fullmatch('[0-9a-f]{64}', h)
                    for h in (self.bag_sha256, self.input_sha256)) or
                type(self.raw_source_keys) is not tuple or len(self.raw_source_keys) != 5 or
                any(type(k) is not str or not k for k in self.raw_source_keys) or
                len(set(self.raw_source_keys)) != 5):
            raise ValueError('exact immutable model-window source binding required')


@dataclass(frozen=True)
class CausalPrefixBinding:
    """Bounded prefix under the explicitly authenticated common ROS clock.

    This is not a claim about the unrecorded capture tail. Inductively output j
    must consume raw j+6: six callbacks are discarded; ordered callbacks emit
    at most once; output j was recorded before raw j+7 was even generated.
    A prefix ends at the FIRST failed bracket, not at a selected passing row.
    This proves transport identity, not registration or physical pose accuracy.
    """
    provenance: AuditedScanProvenance
    clock_evidence_ref: str
    clock_evidence_sha256: str
    # Producer verifies pinned use_sim_time, sensor measurement clock, recorder
    # clock and monotonic /clock records. This pure type cannot authenticate I/O.
    common_clock_contract: str

    def __post_init__(self):
        if not isinstance(self.provenance, AuditedScanProvenance):
            raise ValueError('AUDITED_STARTUP_REQUIRED')
        if (not self.clock_evidence_ref or not re.fullmatch('[0-9a-f]{64}', self.clock_evidence_sha256)
                or self.common_clock_contract != 'pinned_gazebo_measurement_and_rosbag_sim_clock_v1'):
            raise ValueError('AUTHENTICATED_COMMON_CLOCK_REQUIRED')

    @property
    def prefix_count(self):
        count = 0
        for i, output in enumerate(self.provenance.registered):
            # The terminal input has no next-generation upper bound.
            if i + 7 >= len(self.provenance.raw):
                break
            source, following = self.provenance.raw[i+6:i+8]
            if not source.stamp_ns <= source.arrival_ns <= output.arrival_ns < following.stamp_ns:
                break
            count += 1
        return count

    def bind_native(self, registered_source_keys, *, snapshot_stamp_ns, snapshot_receipt_ns):
        keys = tuple(registered_source_keys)
        if len(keys) != 5 or len(set(keys)) != 5:
            raise ValueError('FIVE_DISTINCT_NATIVE_KEYS_REQUIRED')
        certified = {g.stamp_ns: i for i, g in enumerate(self.provenance.registered[:self.prefix_count])}
        indices = []
        for key in keys:
            if not isinstance(key, str) or not key.startswith('registered_scan:'):
                raise ValueError('REGISTERED_SOURCE_KEY_REQUIRED')
            suffix = key.split(':', 1)[1]
            if not suffix.isdecimal() or int(suffix) not in certified:
                raise ValueError('OUTSIDE_CERTIFIED_CAUSAL_PREFIX')
            indices.append(certified[int(suffix)])
        if indices != list(range(indices[0], indices[0]+5)):
            raise ValueError('CONSECUTIVE_ORDERED_NATIVE_WINDOW_REQUIRED')
        pairs = [(self.provenance.raw[i+6], self.provenance.registered[i]) for i in indices]
        if (pairs[-1][1].stamp_ns != snapshot_stamp_ns
                or any(max(r.arrival_ns, g.arrival_ns) > snapshot_receipt_ns for r, g in pairs)):
            raise ValueError('NONCAUSAL_NATIVE_WINDOW')
        return dict(status='CAUSAL_PREFIX_SOURCE_BOUND', registered_source_keys=list(keys),
            raw_source_keys=[f'/velodyne_points:{r.stamp_ns}' for r, _ in pairs],
            raw_orders=[i+6 for i in indices], registered_orders=indices,
            clock_evidence_ref=self.clock_evidence_ref, clock_evidence_sha256=self.clock_evidence_sha256,
            complete_capture_boundary_verified=False, transport_source_bound=True,
            physical_pose_verified=False, registration_verified=False,
            task_identity_verified=False, live_token_availability_verified=False,
            usable_for_native_advice=False)


def audit_startup_sequence(raw, registered, *, segment_id, bag_sha256, image,
                           source_sha256):
    """Fail closed on drift, restart, gaps, output loss or noncausal headers.

Hashes are supplied authentication results, not independently read by this pure
function. Caller must bind the inspected bytes and full header inventory to the
original bag seal. Arbitrary user-supplied strings are not proof of their bytes.
"""
    raw, registered = tuple(raw), tuple(registered)
    _audit_fields(raw, registered, segment_id, bag_sha256, image, source_sha256)
    return AuditedScanProvenance(segment_id, bag_sha256, raw, registered, image,
                                tuple(sorted(source_sha256.items())))


def _audit_fields(raw, registered, segment_id, bag_sha256, image, source_sha256):
    if image != IMAGE or source_sha256 != SOURCE_SHA:
        raise ValueError('UNVERIFIED_SOURCE_CONTRACT')
    if (not isinstance(segment_id, str) or not segment_id or type(bag_sha256) is not str or
            not re.fullmatch('[0-9a-f]{64}', bag_sha256)):
        raise ValueError('source segment and bag hash required')
    if len(raw) < 7 or len(registered) != len(raw) - 6:
        raise ValueError('SOURCE_COUNT_NOT_CONSERVED')
    for seq, caller in ((raw, '/gazebo'), (registered, '/vehicleSimulator')):
        if (any(not isinstance(h, ScanHeader) for h in seq)
                or any(h.seq != i or h.caller != caller for i, h in enumerate(seq))
                or any(a.stamp_ns >= b.stamp_ns or a.arrival_ns > b.arrival_ns
                       for a, b in zip(seq, seq[1:]))):
            raise ValueError('NONUNIQUE_OR_INCOMPLETE_STARTUP_SEQUENCE')
    if any(g.arrival_ns < r.arrival_ns for r, g in zip(raw[6:], registered)):
        raise ValueError('OUTPUT_BEFORE_ITS_INPUT')


def bind_exact_window(provenance, *, segment_id, registered_source_keys,
                      snapshot_stamp_ns, snapshot_receipt_ns, windows):
    """Find exact five-frame identity in a cache, not a close timestamp.

    windows: iterable of ModelWindowSource. No scores or
reference labels enter. Missing/ambiguous matches retain fallback, not a made-up
model observation. Does not validate candidate-route feasibility or token bytes.
"""
    if not isinstance(provenance, AuditedScanProvenance) or segment_id != provenance.segment_id:
        raise ValueError('PROVENANCE_SEGMENT_MISMATCH')
    keys = tuple(registered_source_keys)
    if len(keys) != 5 or len(set(keys)) != 5 or any(type(k) is not str for k in keys):
        raise ValueError('exactly five distinct registered frame keys required')
    if any(type(n) is not int or n < 0 for n in (snapshot_stamp_ns, snapshot_receipt_ns)):
        raise ValueError('snapshot sensor stamp and bag receipt time required')
    mapping = {f'registered_scan:{g.stamp_ns}': (f'/velodyne_points:{r.stamp_ns}', r, g)
               for r, g in zip(provenance.raw[6:], provenance.registered)}
    if any(k not in mapping for k in keys):
        return dict(status='UNBOUND', reason='UNMAPPED_REGISTERED_FRAME', model_window_id=None)
    values = [mapping[k] for k in keys]
    if (any(a[1].stamp_ns >= b[1].stamp_ns for a, b in zip(values, values[1:])) or
            values[-1][2].stamp_ns != snapshot_stamp_ns or
            any(max(v[1].arrival_ns, v[2].arrival_ns) > snapshot_receipt_ns for v in values)):
        raise ValueError('NONCAUSAL_RAW_WINDOW')
    raw_keys = tuple(x[0] for x in values)
    identifiers = set(); matching = []
    for window in windows:
        if not isinstance(window, ModelWindowSource):
            raise ValueError('typed source-bound model cache entry required')
        window_id, sources = window.window_id, window.raw_source_keys
        if window_id in identifiers:
            raise ValueError('unique model window identifiers required')
        identifiers.add(window_id)
        if window.segment_id != segment_id or window.bag_sha256 != provenance.bag_sha256:
            raise ValueError('MODEL_CACHE_SOURCE_MISMATCH')
        if tuple(sources) == raw_keys:
            matching.append(window)
    return dict(status='EXACT_CACHE_MATCH_PENDING_CAPTURE_BOUNDARY' if len(matching) == 1 else 'UNBOUND',
                reason=None if len(matching) == 1 else ('NO_EXACT_MODEL_WINDOW' if not matching else 'AMBIGUOUS_MODEL_WINDOW'),
                model_window_id=matching[0].window_id if len(matching) == 1 else None,
                input_sha256=matching[0].input_sha256 if len(matching) == 1 else None,
                raw_source_keys=list(raw_keys), registered_source_keys=list(keys),
                source_segment=segment_id, bag_sha256=provenance.bag_sha256,
                snapshot_sensor_stamp_ns=snapshot_stamp_ns, snapshot_receipt_ns=snapshot_receipt_ns,
                registration_verified=False, task_identity_verified=False,
                live_token_availability_verified=False,
                capture_boundary_verified=False, usable_for_native_advice=False)
