"""ROS-free sidecar decisions; no model inference or waypoint publication."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from mtare_topo.integration.native_route_advice import decode_advice, make_route_advice, validate_route_advice, validate_snapshot


class PinnedAdviceCache:
    """Read-only, SHA-pinned exact-epoch advice. A digest is not model validity.

    A run-approved producer must establish geometry/model qualification. This
    loader only authenticates declared artifact bytes and the response binding.
    Static advice from a different episode is never rebased to a new epoch.
    """
    def __init__(self, manifest_path, expected_sha256):
        path = Path(manifest_path).resolve()
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('CACHE_MANIFEST_SHA_MISMATCH')
        manifest = decode_advice(raw.decode())
        if manifest.get('schema_version') != 'native_route_advice_cache_v1':
            raise ValueError('CACHE_SCHEMA')
        self.entries = {}
        for entry in manifest['entries']:
            epoch = entry['epoch']
            if not isinstance(epoch, str) or not epoch or epoch in self.entries:
                raise ValueError('CACHE_EPOCH')
            target = (path.parent / entry['path']).resolve()
            if target == path.parent or path.parent not in target.parents:
                raise ValueError('CACHE_PATH_ESCAPE')
            payload = target.read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry['sha256']:
                raise ValueError('CACHE_ADVICE_SHA_MISMATCH')
            advice = decode_advice(payload.decode())
            if advice.get('epoch') != epoch:
                raise ValueError('CACHE_ENTRY_EPOCH_MISMATCH')
            self.entries[epoch] = advice
        self.manifest_sha256 = expected_sha256

    def get(self, epoch):
        return deepcopy(self.entries.get(epoch))


def reply_for_snapshot(snapshot, *, now_ns, cache=None):
    validate_snapshot(snapshot)
    identity = make_route_advice(snapshot, snapshot['original_route'], evidence_refs=[])
    advice = cache.get(snapshot['epoch']) if cache is not None else None
    reason = 'NO_MODEL_CACHE' if cache is None else 'EXACT_EPOCH_NOT_IN_CACHE'
    cache_decision = None
    if advice is not None:
        cache_decision = validate_route_advice(snapshot, advice, now_ns=now_ns)
        if cache_decision['accepted']:
            reason = 'PINNED_EXACT_EPOCH_CACHE'
        else:
            reason = 'CACHE_REJECTED:' + cache_decision['reason']
            advice = None
    if advice is None:
        advice = identity
    decision = validate_route_advice(snapshot, advice, now_ns=now_ns)
    # Even an identity response must be causal; stale epochs receive no reply.
    return dict(reason=reason, advice=advice if decision['accepted'] else None,
        decision=decision, cache_decision=cache_decision,
        model_executed=False, waypoint_published=False,
        identity_fallback=advice == identity)
