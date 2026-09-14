"""Split each upstream label into induced undirected connected components.

No new edges, distances, filtering or semantic labels. Geometric adjacency is
not traversability. IDs are deterministic for fixed point order, not places.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConnectedPartition:
    raw_assignment: np.ndarray
    point_to_component: np.ndarray
    component_to_raw: np.ndarray
    components: tuple


def split_connected_partition(raw_assignment, source, target):
    for value in (raw_assignment, source, target):
        if (not isinstance(value, np.ndarray) or value.ndim != 1
                or value.dtype.kind not in 'iu'):
            raise ValueError('one-dimensional integer arrays required')
    n = len(raw_assignment)
    if source.shape != target.shape or np.any(raw_assignment < 0):
        raise ValueError('edge shapes or negative raw labels')
    if any(np.any(x < 0) or np.any(x >= n) for x in (source, target)):
        raise ValueError('edge index outside input points')
    parent = np.arange(n, dtype=np.int64)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for u, v in zip(source, target):
        u, v = int(u), int(v)
        if raw_assignment[u] != raw_assignment[v]:
            continue
        a, b = find(u), find(v)
        parent[max(a, b)] = min(a, b)
    buckets = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    groups = tuple(np.array(g, dtype=np.int64) for g in buckets.values())
    output = np.empty(n, dtype=np.int64)
    for k, group in enumerate(groups):
        output[group] = k
    raw = raw_assignment.copy()
    mapping = np.array([raw[g[0]] for g in groups], dtype=raw.dtype)
    for array in (raw, output, mapping, *groups):
        array.flags.writeable = False
    return ConnectedPartition(raw, output, mapping, groups)
