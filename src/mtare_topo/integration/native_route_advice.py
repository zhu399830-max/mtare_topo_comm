"""Non-destructive advice for an already feasible native M-TARE route.

This contract does not generate candidates, alter connectivity, publish a
waypoint, or certify learned semantics. All costs are native integer distance
matrix units (including native offsets), deliberately not labelled metres.
"""
from collections import Counter
from copy import deepcopy
import json


SNAPSHOT_SCHEMA = "native_route_snapshot_v1"
ADVICE_SCHEMA = "native_route_advice_v1"
ADVICE_FIELDS = frozenset(("schema_version", "epoch", "snapshot_stamp_ns",
    "source_frame_keys", "candidate_ids", "route", "claimed_cost_units", "evidence_refs"))
I64_MAX = (1 << 63) - 1


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= I64_MAX:
        raise ValueError(name)
    return value


def _ids(values, name, unique=False):
    if not isinstance(values, list):
        raise ValueError(name)
    for value in values:
        _integer(value, name)
    if unique and len(set(values)) != len(values):
        raise ValueError(name)
    return values


def _sources(values, name, allow_empty=False):
    if (not isinstance(values, list) or (not values and not allow_empty)
            or any(not isinstance(v, str) or not v or len(v) > 512 for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(name)
    return values


def route_cost(route, edge_costs):
    """Use only supplied native directed edges; absence means unreachable."""
    total = 0
    for first, second in zip(route, route[1:]):
        if (first, second) not in edge_costs:
            raise ValueError("UNREACHABLE_NATIVE_EDGE")
        total += edge_costs[first, second]
        if total > I64_MAX:
            raise ValueError("COST_OVERFLOW")
    return total


def validate_snapshot(snapshot):
    """Validate a trusted native snapshot; return its directed cost lookup."""
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError("SNAPSHOT_SCHEMA")
    if not isinstance(snapshot.get("epoch"), str) or not snapshot["epoch"]:
        raise ValueError("SNAPSHOT_EPOCH")
    _integer(snapshot["stamp_ns"], "SNAPSHOT_STAMP")
    _integer(snapshot["max_age_ns"], "MAX_AGE", 1)
    _integer(snapshot["max_extra_cost_units"], "MAX_EXTRA_COST")
    _sources(snapshot["source_frame_keys"], "SNAPSHOT_SOURCES")
    _ids(snapshot["candidate_ids"], "SNAPSHOT_CANDIDATES", unique=True)
    route = _ids(snapshot["original_route"], "SNAPSHOT_ROUTE")
    if len(route) < 2 or route[0] != route[-1]:
        raise ValueError("SNAPSHOT_ENDPOINTS")
    if any(v not in snapshot["candidate_ids"] and v != route[0] for v in route):
        raise ValueError("SNAPSHOT_ROUTE_MEMBERS")
    costs = {}
    for edge in snapshot["native_edges"]:
        if not isinstance(edge, list) or len(edge) != 3:
            raise ValueError("SNAPSHOT_EDGE")
        first, second, cost = (_integer(v, "SNAPSHOT_EDGE") for v in edge)
        if (first, second) in costs:
            raise ValueError("DUPLICATE_NATIVE_EDGE")
        costs[first, second] = cost
    expected = _integer(snapshot["original_cost_units"], "ORIGINAL_COST")
    if route_cost(route, costs) != expected:
        raise ValueError("ORIGINAL_COST_MISMATCH")
    return costs


def validate_route_advice(snapshot, advice, *, now_ns):
    """Return a copy of the exact original route on every advice rejection.

    Invalid native snapshots raise: callers must preserve the native route and
    report that integration defect, never repair its data or cost matrix.
    """
    costs = validate_snapshot(snapshot)
    result = dict(schema_version="native_route_decision_v1", epoch=snapshot["epoch"],
        accepted=False, changed=False, reason="NO_ADVICE", route=deepcopy(snapshot["original_route"]),
        original_cost_units=snapshot["original_cost_units"], advised_cost_units=None,
        native_candidates_preserved=True, native_edges_changed=False)
    if advice is None:
        return result
    try:
        _integer(now_ns, "NOW_STAMP")
        if not isinstance(advice, dict) or set(advice) != ADVICE_FIELDS:
            raise ValueError("ADVICE_FIELDS")
        if advice["schema_version"] != ADVICE_SCHEMA:
            raise ValueError("ADVICE_SCHEMA")
        if advice["epoch"] != snapshot["epoch"]:
            raise ValueError("EPOCH_MISMATCH")
        _integer(advice["snapshot_stamp_ns"], "ADVICE_STAMP")
        if advice["snapshot_stamp_ns"] != snapshot["stamp_ns"]:
            raise ValueError("STAMP_MISMATCH")
        age = now_ns - snapshot["stamp_ns"]
        if age < 0 or age > snapshot["max_age_ns"]:
            raise ValueError("ADVICE_NOT_FRESH")
        _sources(advice["source_frame_keys"], "ADVICE_SOURCES")
        if advice["source_frame_keys"] != snapshot["source_frame_keys"]:
            raise ValueError("SOURCE_MISMATCH")
        _ids(advice["candidate_ids"], "ADVICE_CANDIDATES", unique=True)
        if sorted(advice["candidate_ids"]) != sorted(snapshot["candidate_ids"]):
            raise ValueError("CANDIDATE_SET_MISMATCH")
        route = _ids(advice["route"], "ADVICE_ROUTE")
        original = snapshot["original_route"]
        if len(route) < 2 or (route[0], route[-1]) != (original[0], original[-1]):
            raise ValueError("ENDPOINT_MISMATCH")
        if Counter(route) != Counter(original):
            raise ValueError("ROUTE_MULTISET_MISMATCH")
        cost = route_cost(route, costs)
        claimed = _integer(advice["claimed_cost_units"], "CLAIMED_COST")
        if cost != claimed:
            raise ValueError("ADVISED_COST_MISMATCH")
        if cost > snapshot["original_cost_units"] + snapshot["max_extra_cost_units"]:
            raise ValueError("COST_BUDGET_EXCEEDED")
        changed = route != original
        _sources(advice["evidence_refs"], "ADVICE_EVIDENCE", allow_empty=not changed)
        result.update(accepted=True, changed=changed, reason="ACCEPTED" if changed else "IDENTITY_ADVICE",
            route=list(route), advised_cost_units=cost)
    except (ValueError, KeyError, TypeError) as error:
        result["reason"] = str(error)
    return result


def make_route_advice(snapshot, route, *, evidence_refs):
    """Construct advice without certifying it; validator remains authoritative."""
    costs = validate_snapshot(snapshot)
    return dict(schema_version=ADVICE_SCHEMA, epoch=snapshot["epoch"],
        snapshot_stamp_ns=snapshot["stamp_ns"], source_frame_keys=list(snapshot["source_frame_keys"]),
        candidate_ids=list(snapshot["candidate_ids"]), route=list(route),
        claimed_cost_units=route_cost(route, costs), evidence_refs=list(evidence_refs))


def decode_advice(message):
    """Reject duplicate JSON keys and oversized transport before validation."""
    if not isinstance(message, str) or len(message.encode("utf-8")) > 1048576:
        raise ValueError("ADVICE_MESSAGE_SIZE")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    return json.loads(message, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))
