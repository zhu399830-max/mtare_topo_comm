"""Exact metadata-only quota assignment; no I/O, scores, labels or sampling.

Source -> (split,stratum) quotas -> optional structure-node capacity -> host
edge capacity -> parent capacity levels -> sink. Corridor/alias bypass structure nodes. A node's stratum
must be unique, making this network exactly the declared conflict problem.

Maximum cardinality comes first. Among maximum flows, maximize the number of
parents with >=1 selection, then >=2, and so on; finally exact integer binary
preferences select the lexicographically earliest SHA-ranked nomination set.
No floating epsilon, solver tolerance, or heuristic shortage is involved.
"""
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json


SEED = 20260906
STRATA = ("junction", "terminal", "corridor", "alias")
QUOTAS = {"fit": 8, "calibration": 2, "development": 2}


@dataclass(frozen=True)
class ReviewNomination:
    parent: str
    split: str
    host_physical_edge_id: str
    stratum: str
    target_identity: str
    clip_id: str


@dataclass(frozen=True)
class ReviewQuotaAssignment:
    complete: bool
    selected: tuple[ReviewNomination, ...]
    deficits: tuple[dict, ...]
    counts: dict
    selection_sha256: str
    metadata: dict


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _rank(nomination):
    identity = _canonical(asdict(nomination))
    digest = hashlib.sha256(_canonical([SEED, identity]).encode()).hexdigest()
    return digest, identity  # Exact identity is the deterministic hash-collision tie break.


def _validate(nominations):
    if type(nominations) is not tuple:
        raise ValueError("immutable nomination tuple required")
    seen, parents, node_strata, clips = set(), {}, {}, {}
    for n in nominations:
        if type(n) is not ReviewNomination:
            raise ValueError("strict typed metadata nomination required")
        if any(type(v) is not str or not v or v != v.strip() for v in asdict(n).values()):
            raise ValueError("nonempty whitespace-normalized string identities required")
        if n.split not in QUOTAS or n.stratum not in STRATA:
            raise ValueError("unknown split or stratum")
        if n in seen:
            raise ValueError("duplicate nomination")
        seen.add(n)
        if parents.setdefault(n.parent, n.split) != n.split:
            raise ValueError("same parent cannot cross frozen splits")
        if clips.setdefault(n.clip_id, (n.parent, n.host_physical_edge_id)) != (n.parent, n.host_physical_edge_id):
            raise ValueError("same clip claims different parent or physical host edge")
        if n.stratum in ("junction", "terminal"):
            if node_strata.setdefault((n.parent, n.target_identity), n.stratum) != n.stratum:
                raise ValueError("same construction node has conflicting strata")
        elif n.target_identity != n.host_physical_edge_id:
            raise ValueError("corridor and alias must share the physical-edge target identity")


def assign_review_quotas(nominations: tuple[ReviewNomination, ...]) -> ReviewQuotaAssignment:
    """Allocate fixed 8/2/2 per stratum with exact global conflict handling.

    Inputs must already bind the frozen parent split, physical identities and
    valid 21-decision clips. This pure allocator cannot authenticate those
    claims or prove visibility/statistical independence. Shortages are results.

    Integer capacities make every solution integral. Each feasible nomination
    set is a network flow and vice versa; augmenting to exhaustion therefore
    proves maximal cardinality. Exact minimum-cost augmentations optimize the
    coverage profile and finally the unique binary nomination preference.
    With F<=48, running time is O(N log N + F E log V) bigint operations,
    where integer bit length is N+O(32 log 49); no exponential subset search.
    """
    _validate(nominations)
    ordered = tuple(sorted(nominations, key=_rank))
    count = len(ordered)
    quotas = [(split, stratum) for split in QUOTAS for stratum in STRATA]
    node_keys = sorted({(n.parent, n.target_identity) for n in ordered if n.stratum in ("junction", "terminal")})
    host_keys = sorted({(n.parent, n.host_physical_edge_id) for n in ordered})
    parent_split = {n.parent: n.split for n in ordered}
    parent_keys = sorted(parent_split)
    requested_total = sum(QUOTAS.values()) * len(STRATA)
    maximum_level = max(QUOTAS.values()) * len(STRATA)
    # Node-in/out split gives capacity one while retaining all incident-edge
    # alternatives; allocating a nomination never commits its teacher label.
    q_vertex = {key: i + 1 for i, key in enumerate(quotas)}
    node_in = {key: len(q_vertex) + 1 + 2 * i for i, key in enumerate(node_keys)}
    node_out = {key: value + 1 for key, value in node_in.items()}
    host_vertex = {key: len(q_vertex) + 2 * len(node_keys) + 1 + i for i, key in enumerate(host_keys)}
    parent_vertex = {key: len(q_vertex) + 2 * len(node_keys) + len(host_keys) + 1 + i
                     for i, key in enumerate(parent_keys)}
    sink = len(q_vertex) + 2 * len(node_keys) + len(host_keys) + len(parent_keys) + 1
    graph = [[] for _ in range(sink + 1)]

    def add(u, v, capacity, cost=0):
        forward = [v, len(graph[v]), capacity, cost]
        reverse = [u, len(graph[u]), 0, -cost]
        graph[u].append(forward); graph[v].append(reverse)
        return forward

    for key, vertex in q_vertex.items():
        add(0, vertex, QUOTAS[key[0]])
    node_kind = {(n.parent, n.target_identity): (n.split, n.stratum)
                 for n in ordered if n.stratum in ("junction", "terminal")}
    for key in node_keys:
        add(q_vertex[node_kind[key]], node_in[key], 1)
        add(node_in[key], node_out[key], 1)
    for (parent, _), vertex in host_vertex.items():
        add(vertex, parent_vertex[parent], 1)
    # At most requested_total parents can appear at any level, so base total+1
    # is an exact lexicographic encoding of the coverage profile. Shifting N
    # bits makes even its lowest unit dominate ALL nomination preference bits.
    # No quotas, physical thresholds or class decisions enter this tie-break.
    for parent in parent_keys:
        for level in range(1, QUOTAS[parent_split[parent]] * len(STRATA) + 1):
            reward = ((requested_total + 1) ** (maximum_level - level)) << count
            add(parent_vertex[parent], sink, 1, -reward)
    choices = []
    for rank, n in enumerate(ordered):
        origin = (node_out[n.parent, n.target_identity] if n.stratum in ("junction", "terminal")
                  else q_vertex[n.split, n.stratum])
        # Binary integer weights, NOT floating epsilon. One earlier bit is
        # strictly greater than the sum of every later bit, for any N.
        choices.append(add(origin, host_vertex[n.parent, n.host_physical_edge_id], 1,
                           -(1 << (count - rank - 1))))

    # Initial positive-capacity graph is a DAG in vertex-number order. Its
    # exact shortest distances provide feasible potentials for Dijkstra.
    potentials = [None] * len(graph); potentials[0] = 0
    for u in range(len(graph)):
        if potentials[u] is None:
            continue
        for v, _, capacity, cost in graph[u]:
            if capacity and (potentials[v] is None or potentials[u] + cost < potentials[v]):
                potentials[v] = potentials[u] + cost
    potentials = [0 if p is None else p for p in potentials]
    flow = 0
    while True:
        distances = [None] * len(graph); predecessor = [None] * len(graph)
        distances[0] = 0; queue = [(0, 0)]
        while queue:
            distance, u = heapq.heappop(queue)
            if distances[u] != distance:
                continue
            for edge_index, (v, _, capacity, cost) in enumerate(graph[u]):
                if not capacity:
                    continue
                reduced = cost + potentials[u] - potentials[v]
                if reduced < 0:
                    raise RuntimeError("exact shortest-path potential invariant violated")
                candidate = distance + reduced
                if distances[v] is None or candidate < distances[v]:
                    distances[v] = candidate; predecessor[v] = (u, edge_index)
                    heapq.heappush(queue, (candidate, v))
        if distances[sink] is None:
            break
        for v, distance in enumerate(distances):
            if distance is not None:
                potentials[v] += distance
        v = sink
        while v != 0:
            u, edge_index = predecessor[v]
            edge = graph[u][edge_index]
            edge[2] -= 1; graph[v][edge[1]][2] += 1
            v = u
        flow += 1
    selected = tuple(n for n, edge in zip(ordered, choices) if edge[2] == 0)
    if len(selected) != flow:
        raise RuntimeError("nomination flow cardinality invariant violated")
    counts, deficits = {}, []
    for split, quota in QUOTAS.items():
        counts[split] = {}
        for stratum in STRATA:
            candidates = [n for n in ordered if (n.split, n.stratum) == (split, stratum)]
            actual = sum((n.split, n.stratum) == (split, stratum) for n in selected)
            counts[split][stratum] = {"quota": quota, "selected": actual, "raw_nominations": len(candidates),
                "candidate_host_edges": len({(n.parent, n.host_physical_edge_id) for n in candidates}),
                "candidate_targets": len({(n.parent, n.target_identity) for n in candidates})}
            if actual < quota:
                deficits.append({"split": split, "stratum": stratum, "required": quota,
                                 "selected": actual, "missing": quota - actual})
    selected_identity = [asdict(n) for n in selected]
    parent_counts = {p: sum(n.parent == p for n in selected) for p in parent_keys}
    coverage_profile = [sum(v >= level for v in parent_counts.values())
                        for level in range(1, maximum_level + 1)]
    digest = hashlib.sha256(_canonical([SEED, QUOTAS, selected_identity]).encode()).hexdigest()
    return ReviewQuotaAssignment(not deficits, selected, tuple(deficits), counts, digest, {
        "seed": SEED, "input_nominations": count, "host_edges": len(host_keys), "node_targets": len(node_keys),
        "maximum_cardinality": flow, "requested_total": requested_total,
        "algorithm": "exact_integer_min_cost_max_flow_parent_level_coverage_then_binary_preference",
        "objective_order": ["maximum_cardinality", "lexicographic_parent_coverage_by_level",
                            "lexicographic_seed_hash_nomination_set"],
        "parent_selected_counts": parent_counts, "parent_coverage_profile": coverage_profile,
        "optimality": "no_augmenting_path; shortest_residual_paths_with_exact_integer_potentials",
        "tie_break": "sha256(seed,full_nomination_identity),then_full_identity;exact_binary_integers",
        "input_sha256": hashlib.sha256(_canonical([asdict(n) for n in ordered]).encode()).hexdigest(),
        "network_vertices": len(graph), "network_forward_edges": sum(map(len, graph)) // 2,
        "source_provenance_verified": False, "human_labels_created": 0, "training_eligibility": False,
        "statistically_independent_unit": "parent_world_not_clip_or_nomination",
    })
