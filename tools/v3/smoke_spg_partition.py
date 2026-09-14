"""Software-only fixed128-point smoke of separately built official libraries.

No dataset, labels, training, Delaunay graph or voxel preprocessing. This is
not the full R2 extractor; native calls receive validated finite arrays.
"""
import json
import _bootstrap
import time
import numpy as np
import libply_c
import libcp
from mtare_topo.representation.gse_connected_partition import split_connected_partition


def main():
    rng = np.random.default_rng(20260906)
    xy = rng.uniform(-1, 1, (64, 2))
    points = np.concatenate([np.column_stack([xy, np.zeros(64)]),
                             np.column_stack([xy, np.ones(64)*3])]).astype('float32')
    distance = np.linalg.norm(points[:, None]-points[None], axis=-1)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1, kind='stable')[:, :45].astype('uint32')
    geof = libply_c.compute_geof(points, np.ascontiguousarray(neighbors.ravel()), 45)
    features = np.ascontiguousarray(geof, dtype='float32')
    assert features.shape == (128, 4) and np.isfinite(features).all()
    features[:, 3] *= 2
    source = np.repeat(np.arange(128, dtype='uint32'), 10)
    target = np.ascontiguousarray(neighbors[:, :10].ravel())
    cross_layer_edges = int(((source < 64) != (target < 64)).sum())
    adjacency = [set() for _ in points]
    for u, v in zip(source, target):
        adjacency[u].add(int(v)); adjacency[v].add(int(u))
    unseen = set(range(len(points))); graph_sizes = []
    while unseen:
        todo = [min(unseen)]; reached = set()
        while todo:
            u = todo.pop()
            if u in reached:
                continue
            reached.add(u); todo.extend(adjacency[u] - reached)
        unseen -= reached; graph_sizes.append(len(reached))
    distances = distance[source, target]
    weights = np.ascontiguousarray(1/(1+distances/distances.mean()), dtype='float32')
    rows = []; assignments = []
    for _ in range(3):
        start = time.monotonic()
        components, assignment = libcp.cutpursuit(features, source, target, weights, .1)
        assignment = np.asarray(assignment)
        flattened = np.concatenate([np.asarray(c) for c in components])
        assert assignment.shape == (128,)
        assert np.array_equal(np.sort(flattened), np.arange(128))
        for i, component in enumerate(components):
            assert np.all(assignment[np.asarray(component)] == i)
        assignments.append(assignment.copy())
        refined = split_connected_partition(assignment, source, target)
        assert sorted(map(len, refined.components)) == [64,64]
        rows.append(dict(components=len(components), sizes=[len(c) for c in components],
                         connected_sizes=[len(c) for c in refined.components],
                         elapsed_s=time.monotonic()-start))
    same = all(np.array_equal(assignments[0][:,None]==assignments[0][None,:],
                              a[:,None]==a[None,:]) for a in assignments[1:])
    print(json.dumps(dict(points=128, geometry_neighbors=45, adjacency_neighbors=10,
        cross_layer_edges=cross_layer_edges, input_connected_component_sizes=graph_sizes,
        repeats=rows, equivalent_partitions=same, all_points_accounted=True,
        full_extractor_qualified=False, scientific_gate_pass=False, optimizer_steps=0)))


if __name__ == '__main__':
    main()
